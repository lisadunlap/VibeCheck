import argparse
import wandb
import pandas as pd
from plotly import graph_objects as go
import numpy as np
import os
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.cache import CacheConfig, CacheType, CacheFactory

from typing import List

from utils import (
    proposer_postprocess,
    parse_axes,
    get_pref_score,
    train_and_evaluate_model,
    get_feature_df,
    ranker_postprocess,
    create_side_by_side_plot,
)


def rank_axes(vibes: List[str], df: pd.DataFrame, models: List[str], single_position_rank: bool = False):
    """
    Ranks the two model outputs across the given vibes (axes) using LOTUS ranking prompts.
    """
    judge_systems_prompt = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a given propoery. Which repose better aligns more with the given property, A, B, or equal?

Your sole focus is to determine which response better aligns with the given property, NOT how good or bad the response is. Avoid any position bias and remain as objective as possible. Consider what the property means and how it applies to the outputs. Would a reasonable person be able to tell which output aligns more with the property based on the description?

Instructions:
	•	If Response A aligns with the property more than Response B, respond with "A".
    •	If Response B aligns with the property more than Response A, respond with "B".
	•	If the responses are roughly equal on the property, respond with "equal".
	•	If the property does not apply to these outputs (e.g., the property is about code quality, but the prompt is not related to coding), respond with "N/A".
	•	If you are unsure about the meaning of the property, respond with "unsure". Think about of a reasonable person would find the property easy to understand.

A group of humans should agree with your decision. Use the following format for your response:
Explanation: {{your explanation}}
Text from outputs which aligns with the property: "{{text from outputs which aligns with the property}}"
Text from outputs which does not align with the property: "{{text from outputs which does not align with the property}}"
Model: {{A, B, equal, N/A, or unsure}}

Remember to be as objective as possible and strictly adhere to the response format."""

    ranker_prompt1 = (
        judge_systems_prompt
        + """
Here is the property and the two responses:
{ranker_inputs}

Remember to be as objective as possible and strictly adhere to the response format.
"""
    )

    ranker_prompt2 = (
        judge_systems_prompt
        + """
Here is the property and the two responses:
{ranker_inputs_reversed}

Remember to be as objective as possible and strictly adhere to the response format.
"""
    )

    vibe_dfs = []
    for vibe in vibes:
        vibe_df = df.copy()
        vibe_df["vibe"] = vibe
        vibe_dfs.append(vibe_df)

    vibe_df = pd.concat(vibe_dfs).reset_index(drop=True)

    # drop any duplicate columns
    vibe_df = vibe_df.loc[:, ~vibe_df.columns.duplicated()]
    vibe_df["ranker_inputs"] = vibe_df.apply(
        lambda row: f"\nProperty: {row['vibe']}\n\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[models[0]]}\n\nResponse B:\n{row[models[1]]}\n\nProperty (restated): {row['vibe']}",
        axis=1,
    )
    if not single_position_rank:
        vibe_df["ranker_inputs_reversed"] = vibe_df.apply(
            lambda row: f"Property: {row['vibe']}\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[models[1]]}\n\nResponse B:\n{row[models[0]]}\n\nProperty (restated): {row['vibe']}",
            axis=1,
        )

    ranker_1 = vibe_df.sem_map(
        ranker_prompt1, return_raw_outputs=True, suffix="ranker_output_1"
    )

    vibe_df = vibe_df.merge(
        ranker_1[
            [
                "vibe",
                "question",
                models[0],
                models[1],
                "preference",
                "ranker_output_1",
                "raw_outputranker_output_1",
            ]
        ],
        on=["vibe", "question", models[0], models[1], "preference"],
        how="left",
    )
    vibe_df["ranker_output_1"] = vibe_df["ranker_output_1"].apply(ranker_postprocess)
    if not single_position_rank:
        ranker_2 = vibe_df.sem_map(
            ranker_prompt2, return_raw_outputs=True, suffix="ranker_output_2"
        )
        vibe_df = vibe_df.merge(
            ranker_2[
                ["question", models[0], models[1], "preference", "ranker_output_2"]
            ],
            on=["question", models[0], models[1], "preference"],
            how="left",
        )
        vibe_df["ranker_output_2"] = vibe_df["ranker_output_2"].apply(
            ranker_postprocess
        )
        vibe_df["position_matters"] = (
            vibe_df["ranker_output_1"] != -1 * vibe_df["ranker_output_2"]
        )
        vibe_df["score"] = vibe_df.apply(
            lambda row: row["ranker_output_1"] if not row["position_matters"] else 0,
            axis=1,
        )
        wandb.summary["prop_position_collisions"] = vibe_df["position_matters"].mean()
    else:
        vibe_df["score"] = vibe_df["ranker_output_1"]

    return vibe_df


def create_reduce_prompt(num_reduced_axes: int):
    return f"""Below is a list of properties that are found in LLM outputs. I would like to summarize this list to AT MOST {num_reduced_axes} representative properties with concise descriptions. Are there any overarching properties that are present in a large number of the properties?

Here is the list of properties:
{{differences}}

Your final list of simplified properties should be human interpretable. The final list of descriptions should be unambiguous and concise. For example, 
* "uses a lot of emojis and markdown" is not a good property because a piece of text can have emojies but not markdown, and vice versa. This should be split into two properties: "uses a lot of emojis" and "uses markdown".
* if two properties are "uses markdown" and "utilizes extensive formatting", text which contains one likely contains the other and should be combined into a single property "uses extensive markdown formatting".
* "focus on historical context" is not a good property because it is too vague. A better property would be "mentions specific historical events".

Each property should be <= 10 words. Order your final list of properties by how much they are seen in the data. Your response should be a list deliniated with "-"
"""


def propose_vibes(
    df: pd.DataFrame, models: List[str], num_proposal_samples: int = 30, num_final_vibes: int = 10, batch_size: int = 5
):
    proposer_prompt_freeform = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. Write down as many differences as you can find between the two outputs. Please format your differences as a list of properties that appear more in one output than the other.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, analyze the differences between Model 1 and Model 2. What properties are seen in the responses from Model 1 that are not seen in the responses from Model 2? What properties are seen in the responses from Model 2 that are not seen in the responses from Model 1?

{combined_responses}

The format should be a list of properties that appear more in one output than the other in the format of a short description of the property. An example of a possible output is,
- "conversational language"
- "friendly tone"
- "code that optimizes for runtime"
- "uses a lot of emojis"
- "stories presented in the third person"

Note that this example is not at all exhaustive, but rather just an example of the format. Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. 
    
Remember that these properties should be human interpretable and that the differences should be concise (<= 10 words), substantive and objective. Write down as many properties as you can find. Do not explain which model has which property, simply describe the property.
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""
    # Create combined responses to get in LOTUS format
    df["single_combined_response"] = df.apply(
        lambda row: (
            f"User prompt:\n{row['question']}\n\n"
            f"Model 1:\n{row[models[0]]}\n\n"
            f"Model 2:\n{row[models[1]]}"
        ),
        axis=1,
    )
    proposer_df = df.sample(num_proposal_samples, random_state=42).reset_index(
        drop=True
    )
    proposer_df["batch_id"] = proposer_df.index // batch_size
    proposer_df["combined_responses"] = proposer_df.groupby("batch_id")[
        "single_combined_response"
    ].transform(lambda x: "\n-------------\n".join(x))
    proposer_df = proposer_df.drop_duplicates("batch_id")
    proposer_df = proposer_df.sem_map(
        proposer_prompt_freeform, return_raw_outputs=True, suffix="differences"
    )

    proposer_df["differences"] = proposer_df["differences"].apply(proposer_postprocess)
    wandb.log({"Vibe Proposer/proposer_results": wandb.Table(dataframe=proposer_df)})
    results = proposer_df[proposer_df["differences"].apply(lambda x: len(x) > 0)]
    results = results.explode("differences").reset_index(drop=True)

    # Cluster and reduce axes
    # TODO: fix groupby_clusterid for sem_agg
    results = results.sem_index("differences", "differences_index").sem_cluster_by(
        "differences", 1
    )
    summaries = results.sem_agg(
        create_reduce_prompt(num_final_vibes),
        suffix="reduced axes",
    )
    summaries["reduced axes parsed"] = summaries["reduced axes"].apply(parse_axes)
    vibes = summaries.explode("reduced axes parsed")["reduced axes parsed"].to_list()
    print("Vibes:\n" + "\n".join(vibes))
    return vibes

def get_vibe_question_types(vibe_df: pd.DataFrame, batch_size: int = 50):
    """Describe what types of questions result in high scores for a given vibe."""
    # Create a copy of the filtered dataframe to avoid warnings
    filtered_vibe_df = vibe_df[vibe_df["score"].abs() > 0.0].copy()
    vibin_questions = filtered_vibe_df.sort_values(["vibe", "score"], ascending=[True, False])

    prompt = """You are a machine learning researcher trying to discover what types of questions result in a model exhibiting a certain behavior. Given the following behavior and a list of questions along with a score of how much the model exhibits the behavior on that question, describe what types of questions result in the model exhibiting the behavior. Each score is between 1 and 5, where 1 is does not exhibit the behavior at all, and 5 is exhibits the behavior completely.

{input_text}

Please respond with a 1 sentence description of what types of questions result in the model exhibiting the behavior and what types of questions result in the model not exhibiting the behavior. This should be a single sentence that is human interpretable and concise, such that I can create a diverse set of new questions that result in the model exhibiting the behavior as well as new questions that result in the model not exhibiting the behavior.

Format your response in the following format:
Question types which exhibit the behavior: <description>
Question types which do not exhibit the behavior: <description>
"""
    new_df = []
    for vibe in vibin_questions["vibe"].unique():
        single_vibe_df = vibin_questions[vibin_questions["vibe"] == vibe].copy()
        # Sample and create input text for each vibe
        sampled_df = single_vibe_df.sample(min(batch_size, len(single_vibe_df)))
        input_texts = sampled_df.apply(
            lambda row: f"Behavior: {row['vibe']}\nQuestion: {row['question']}\nScore: {row['score']}",
            axis=1
        ).tolist()
        input_text = "\n-------------\n".join(input_texts)
        new_df.append({
            "vibe": vibe,
            "input_text": input_text
        })
    
    new_df = pd.DataFrame(new_df)
    vibe_question_types = new_df.sem_map(
        prompt, return_raw_outputs=True, suffix="vibe_question_types"
    )
    return vibe_question_types


def get_examples_for_vibe(vibe_df: pd.DataFrame, vibe: str, models: List[str], num_examples: int = 5):
    """Get example pairs where the given vibe was strongly present."""
    vibe_examples = vibe_df[(vibe_df["vibe"] == vibe) & (vibe_df["score"].abs() > 0.0)]
    examples = []
    for _, row in vibe_examples.head(num_examples).iterrows():
        examples.append(
            {
                "prompt": row["question"],
                "output_a": row[models[0]],
                "output_b": row[models[1]],
                "score": row["score"],
                "core_output": row["raw_outputranker_output_1"],
            }
        )
    return examples


def create_gradio_app(vibe_df: pd.DataFrame, models: List[str], coef_df: pd.DataFrame, corr_plot: go.Figure, vibe_question_types: pd.DataFrame):
    import gradio as gr

    # Create the plots
    agg_df = (
        vibe_df.groupby("vibe")
        .agg({"pref_score": "mean", "score": "mean"})
        .reset_index()
    )

    # Create plots and convert them to HTML strings
    heuristics_plot = create_side_by_side_plot(
        df=agg_df,
        y_col="vibe",
        x_cols=["score", "pref_score"],
        titles=("Model Identity", "Preference Prediction"),
        main_title="Vibe Heuristics",
        models=models,
    )

    coef_plot = create_side_by_side_plot(
        df=coef_df,
        y_col="vibe",
        x_cols=["coef_modelID", "coef_preference"],
        titles=("Model Identity", "Preference Prediction"),
        main_title="Vibe Model Coefficients",
        models=models,
        error_cols=["coef_std_modelID", "coef_std_preference"],
    )

    def show_examples(vibe):
        examples = get_examples_for_vibe(vibe_df, vibe, models)
        markdown = f"### What sort of prompts elicit the vibe?\n"
        markdown += f"{vibe_question_types[vibe_question_types['vibe'] == vibe]['vibe_question_types'].values[0]}\n\n"
        markdown += "---\n\n"
        for i, ex in enumerate(examples, 1):
            markdown += f"### Example {i} ({models[0] if ex['score'] > 0 else models[1]} vibe)\n"
            markdown += f"**Prompt:**\n{ex['prompt']}\n\n"
            markdown += f"**{models[0]}:**\n{ex['output_a']}\n\n"
            markdown += f"**{models[1]}:**\n{ex['output_b']}\n\n"
            markdown += f"**Ranker Output:**\n{ex['core_output']}\n\n"
            markdown += "---\n\n"
        return markdown

    with gr.Blocks() as app:
        gr.Markdown("# <center>It's all about the ✨vibes✨</center>")

        with gr.Accordion("Plots", open=True):
            with gr.Row():
                gr.Plot(heuristics_plot)

            with gr.Row():
                gr.Plot(coef_plot)

            with gr.Row():
                gr.Plot(corr_plot)

        gr.Markdown("## Vibe Examples")
        vibe_df_w_types = vibe_df.merge(vibe_question_types, on="vibe", how="left")
        vibe_dropdown = gr.Dropdown(
            choices=vibe_df_w_types["vibe"].unique().tolist(),
            label="Select a vibe to see examples",
        )
        examples_output = gr.Markdown()
        vibe_dropdown.change(
            fn=show_examples, inputs=[vibe_dropdown], outputs=[examples_output]
        )

    return app


def create_vibe_correlation_plot(vibe_df: pd.DataFrame, models: List[str]):
    """Creates a correlation matrix plot for vibe scores."""
    # Pivot the dataframe to get vibe scores in columns
    vibe_pivot = vibe_df.pivot_table(
        index=["question", models[0], models[1]], columns="vibe", values="score"
    ).reset_index()

    # Calculate correlation matrix for just the vibe scores
    vibe_cols = vibe_pivot.columns[3:]  # Skip the index columns
    corr_matrix = vibe_pivot[vibe_cols].corr()

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Vibe Score Correlations",
        xaxis_tickangle=-45,
        width=800,
        height=800,
    )

    return fig


def main(
    data_path: str,
    models: List[str],
    num_proposal_samples: int = 30,
    num_final_vibes: int = 10,
    test: bool = False,
    single_position_rank: bool = False,
    project_name: str = "vibecheck",
    proposer_only: bool = False,
    no_holdout_set: bool = False,
    gradio: bool = False,
):
    """Run VibeCheck analysis to identify and analyze behavioral differences between two language models.

    Args:
        data_path (str): Path to CSV file containing model outputs. Must include columns for model responses 
            and a 'preference' column indicating which model output was preferred.
        models (List[str]): List of two model names to compare. These should match the column names in the CSV.
        num_proposal_samples (int, optional): Number of samples to use when proposing vibes. Defaults to 30.
        num_final_vibes (int, optional): Maximum number of vibes to use in final analysis. Defaults to 10.
        test (bool, optional): If True, runs analysis on a small subset of data for testing. Defaults to False.
        single_position_rank (bool, optional): If True, only ranks model outputs in one position order.
            Faster but may introduce position bias. Defaults to False.
        project_name (str, optional): Name of the Weights & Biases project. Defaults to "vibecheck".
        proposer_only (bool, optional): If True, only runs the vibe proposal step without analysis. Defaults to False.
        no_holdout_set (bool, optional): If True, uses all data for training without a test set. Defaults to False.
        gradio (bool, optional): If True, launches a Gradio interface after analysis. Defaults to False.

    Raises:
        ValueError: If 'preference' column is not found in the input CSV file.
    """
    # Initialize wandb
    wandb.init(project=project_name, name=f"{models[0]}_vs_{models[1]}", save_code=True)

    output_dir = f"outputs/{data_path.split('/')[-1].replace('.csv', '')}_{models[0]}_vs_{models[1]}"
    os.makedirs(output_dir, exist_ok=True)
    # Initialize LOTUS
    cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)
    lm = LM(model="gpt-4o", cache=cache)
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    lotus.settings.configure(lm=lm, rm=rm, enable_cache=True)

    # Load and preprocess data
    df = pd.read_csv(data_path)
    if test:
        df = df.sample(100, random_state=42)
    if "preference" not in df.columns:
        raise ValueError(
            "Preference column not found in dataframe. Run get_preference_labels.py first"
        )

    df = df[df["preference"].isin(models)].reset_index(drop=True)

    print(f"Preference Counts: {df['preference'].value_counts().to_dict()}")
    wandb.summary["preference_counts"] = df["preference"].value_counts().to_dict()
    wandb.summary["data_size"] = len(df)

    # Create bar plot of preference distribution
    pref_dist = df["preference"].value_counts()
    fig = go.Figure(
        data=[go.Bar(x=pref_dist.index, y=pref_dist.values, marker_color="#2ecc71")]
    )
    fig.update_layout(
        title="Model Preference Distribution",
        xaxis_title="Model",
        yaxis_title="Count",
        template="plotly_white",
    )
    wandb.log({"preference_distribution": wandb.Html(fig.to_html())})
    fig.write_html(os.path.join(output_dir, "preference_distribution.html"))
    vibes = propose_vibes(
        df,
        models,
        num_proposal_samples=num_proposal_samples,
        num_final_vibes=num_final_vibes,
    )

    # Log vibes to wandb
    vibes_df = pd.DataFrame({"vibes": vibes})
    wandb.log({"vibes": wandb.Table(dataframe=vibes_df)})
    vibes_df.to_csv(os.path.join(output_dir, "vibes.csv"), index=False)

    if proposer_only:
        return

    # Rank axes
    lm = LM(model="gpt-4o-mini", cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=True)
    if test:
        vibe_df = rank_axes(
            vibes[:3], df, models, single_position_rank=single_position_rank
        )
    else:
        vibe_df = rank_axes(
            vibes, df, models, single_position_rank=single_position_rank
        )

    # Compute preference alignment
    vibe_df["preference_feature"] = vibe_df["preference"].apply(
        lambda x: get_pref_score(x, models)
    )
    vibe_df["pref_score"] = vibe_df["score"] * vibe_df["preference_feature"]

    wandb.log({"Vibe Scoring/ranker_results": wandb.Table(dataframe=vibe_df)})
    vibe_df.to_csv(os.path.join(output_dir, "vibe_df.csv"), index=False)

    agg_df = (
        vibe_df.groupby("vibe")
        .agg({"pref_score": "mean", "score": "mean"})
        .reset_index()
    )
    wandb.log({"summary": wandb.Table(dataframe=agg_df)})

    # Get the aggregated data and parse descriptions
    agg_df = (
        vibe_df.groupby("vibe")
        .agg({"pref_score": "mean", "score": "mean"})
        .reset_index()
    )

    # First plot (vibe heuristics)
    model_vibe_scores_plot = create_side_by_side_plot(
        df=agg_df,
        y_col="vibe",
        x_cols=["score", "pref_score"],
        titles=("Model Identity", "Preference Prediction"),
        main_title="Vibe Heuristics",
        models=models,
    )
    wandb.log({"Vibe Plots/model_vibe_scores_plot": wandb.Html(model_vibe_scores_plot.to_html())})
    model_vibe_scores_plot.write_html(os.path.join(output_dir, "model_vibe_scores_plot.html"))
    # Filter out vibes with low separation or preference
    vibe_df = vibe_df[vibe_df["score"].abs() > 0.05]
    vibe_df = vibe_df[vibe_df["pref_score"].abs() > 0.05]
    print(
        f"Retained {len(vibe_df.drop_duplicates('vibe'))} vibes with non-trivial separation/preference."
    )
    print("Remaining vibes:\n" + "\n".join(vibe_df["vibe"].unique()))

    # Train Preference Prediction and Model Identity Classification Models
    feature_df, X_pref, y_pref, y_identity = get_feature_df(vibe_df)
    (
        preference_model,
        preference_coef_df,
        preference_accuracy_test,
        preference_acc_std,
    ) = train_and_evaluate_model(
        X_pref,
        y_pref,
        feature_df.columns,
        split_train_test=not no_holdout_set,
        solver="elasticnet",
    )
    identity_model, identity_coef_df, identity_accuracy_test, identity_acc_std = (
        train_and_evaluate_model(
            X_pref,
            y_identity,
            feature_df.columns,
            split_train_test=not no_holdout_set,
            solver="elasticnet",
        )
    )

    wandb.log(
        {
            "preference_model_test_accuracy": preference_accuracy_test,
            "identity_model_test_accuracy": identity_accuracy_test,
            "preference_model_test_accuracy_std": preference_acc_std,
            "identity_model_test_accuracy_std": identity_acc_std,
        }
    )

    # Merge coefficient data
    coef_df = identity_coef_df.merge(
        preference_coef_df, on="vibe", suffixes=("_modelID", "_preference")
    ).sort_values("coef_preference", ascending=False)

    # Second plot (coefficients)
    coef_plot = create_side_by_side_plot(
        df=coef_df,
        y_col="vibe",
        x_cols=["coef_modelID", "coef_preference"],
        titles=("Model Identity", "Preference Prediction"),
        main_title="Vibe Model Coefficients",
        models=models,
        error_cols=["coef_std_modelID", "coef_std_preference"],
    )
    wandb.log({"Vibe Plots/model_vibe_coef_plot": wandb.Html(coef_plot.to_html())})
    coef_plot.write_html(os.path.join(output_dir, "model_vibe_coef_plot.html"))
    coef_df.to_csv(os.path.join(output_dir, "vibecheck_coefficients.csv"), index=False)

    # Log final data
    wandb.log({"coefficient_data": wandb.Table(dataframe=coef_df)})

    # After creating vibe_df and before training models, add:
    corr_plot = create_vibe_correlation_plot(vibe_df, models)
    wandb.log({"Vibe Scoring/vibe_correlations": wandb.Html(corr_plot.to_html())})
    corr_plot.write_html(os.path.join(output_dir, "vibe_correlations.html"))

    vibe_question_types = get_vibe_question_types(vibe_df)
    wandb.log({"Vibe Scoring/vibe_question_types": wandb.Table(dataframe=vibe_question_types)})
    vibe_question_types.to_csv(os.path.join(output_dir, "vibe_question_types.csv"), index=False)
    # Close wandb run
    wandb.finish()

    if gradio:
        print("\nLaunching Gradio app...")
        app = create_gradio_app(vibe_df, models, coef_df, corr_plot, vibe_question_types)
        app.launch(share=True)

    return {"output_dir": output_dir,
            "model_vibe_scores_plot": model_vibe_scores_plot,
            "score_dist_plot": coef_plot,
            "vibe_question_types": vibe_question_types,
            "vibe_df": vibe_df,
            "agg_df": agg_df,
            "corr_plot": corr_plot,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VibeCheck analysis on model outputs."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV file containing model outputs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to compare",
    )
    parser.add_argument(
        "--num_proposal_samples",
        type=int,
        default=30,
        help="Number of samples to use for proposing vibes",
    )
    parser.add_argument(
        "--project", type=str, default="vibecheck", help="Wandb project name"
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument(
        "--single_position_rank",
        action="store_true",
        help="Don't rerun ranker with different positions, faster but may lead to position bias",
    )
    parser.add_argument(
        "--num_final_vibes",
        type=int,
        default=10,
        help="Number of final vibes to use for analysis",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="elasticnet",
        help="Solver to use for logistic regression (standard, lasso, elasticnet)",
    )
    parser.add_argument(
        "--proposer_only", action="store_true", help="Only run the proposer"
    )
    parser.add_argument("--gradio", action="store_true", help="Run the Gradio app")
    parser.add_argument(
        "--no_holdout_set", action="store_true", help="Don't split into a holdout set"
    )

    args = parser.parse_args()
    main(
        args.data_path,
        args.models,
        args.num_proposal_samples,
        args.num_final_vibes,
        args.test,
        args.single_position_rank,
        args.project,
        args.proposer_only,
        args.no_holdout_set,
        args.gradio,
    )
