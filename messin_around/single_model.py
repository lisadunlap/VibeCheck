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
    get_pref_score,
    train_and_evaluate_model,
    get_feature_df,
    create_side_by_side_plot,
)


def ranker_postprocess(output: str) -> tuple[float, str, str]:
    """
    Extracts the numerical score, explanation, and text references from the ranker's output.

    Args:
        output (str): Raw output from the ranking model containing explanation and score

    Returns:
        tuple[float, str, str]: Tuple containing:
            - float: Extracted score (1-5) or 0.0 if N/A or invalid
            - str: Explanation text
            - str: Referenced text that aligns with the property
    """
    try:
        explanation = ""
        text_refs = ""
        score = 0.0

        # Extract explanation
        if "Explanation:" in output:
            explanation_parts = (
                output.split("Explanation:")[-1].split("Text from outputs")[0].strip()
            )
            explanation = explanation_parts

        # Extract text references
        if "Text from outputs which aligns with the property:" in output:
            text_parts = (
                output.split("Text from outputs which aligns with the property:")[-1]
                .split("Score:")[0]
                .strip()
            )
            text_refs = text_parts

        # Extract score
        if "Score:" not in output:
            return 0.0, explanation, text_refs

        score_text = output.split("Score:")[-1].strip()

        # Handle N/A case
        if "N/A" in score_text:
            return 0.0, explanation, text_refs

        # Extract first number found in the score text
        score = float(next(num for num in score_text.split() if num.isdigit()))

        # Validate score range
        if score < 1 or score > 5:
            print(f"Invalid score: {score} in {output}")
            return 0.0, explanation, text_refs

        return score, explanation, text_refs

    except (ValueError, StopIteration):
        print(f"Error in ranker_postprocess: {output}")
        return 0.0, "", ""


def rank_axes(vibes: List[str], df: pd.DataFrame, model: str):
    """
    Ranks the two model outputs across the given vibes (axes) using LOTUS ranking prompts.
    """
    judge_systems_prompt = """You are a fair and unbiased judge doing qualitative analysis on LLM outputs. Given a model output and property, your task is to judge how well a model's output aligns with a given property. Would a group of humans agree that the model's output has the property? If so, to what extent?

Please respond with a score between 1 and 5, where 1 means the model's output does not align with the property at all, 5 means the model's output aligns with the property completely. If the property is irrelevant to the prompt (e.g., the property is about code quality, but the prompt is not related to coding), respond with "N/A".

For example, if the property is "casual tone", a score of 1 would mean the model's output is very formal, 5 would mean the model's output is very casual, and 4 would mean the model's output is somewhat casual.

A group of humans should agree with your decision. Provide an explanation of your score and reference parts of the model's output that align or do not align with the property to justify your score. 
Explanation: {{your explanation}}
Text from outputs which aligns with the property: "{{text from outputs which aligns with the property, with each quote seperated by newlines and a -}}"
Score: {{1, 2, 3, 4, 5, or N/A}}

Remember to be as objective as possible and strictly adhere to the response format."""

    ranker_prompt1 = (
        judge_systems_prompt
        + """
Here is the property and the two responses:
{ranker_inputs}

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
        lambda row: f"\nProperty: {row['vibe']}\n\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[model]}\n\nProperty (restated): {row['vibe']}",
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
                model,
                "preference",
                "ranker_output_1",
                "raw_outputranker_output_1",
            ]
        ],
        on=["vibe", "question", model, "preference"],
        how="left",
    )
    vibe_df[["score", "explanation", "text_refs"]] = pd.DataFrame(
        vibe_df["ranker_output_1"].apply(ranker_postprocess).tolist(),
        index=vibe_df.index,
    )

    # Create plot with vibes on y-axis and side-by-side bars for score counts
    fig = go.Figure()
    possible_scores = [0, 1, 2, 3, 4, 5]

    for score in possible_scores:
        counts = []
        for vibe in vibe_df["vibe"].unique():
            vibe_df_vibe = vibe_df[vibe_df["vibe"] == vibe]
            count = (vibe_df_vibe["score"] == score).sum()
            counts.append(count)

        fig.add_trace(
            go.Bar(
                name=f"Score {score}",
                y=vibe_df["vibe"].unique(),
                x=counts,
                orientation="h",
            )
        )

    fig.update_layout(
        title="Score Distribution per Vibe<br><sup>Note: 0 indicates vibe is irrelevant to the prompt</sup>",
        yaxis_title="Vibe",
        xaxis_title="Count",
        barmode="group",
        height=800,
    )

    wandb.log({"Vibe Scoring/score_value_counts_plot": wandb.Html(fig.to_html())})
    return vibe_df, fig


def create_reduce_prompt(num_reduced_axes: int):
    return f"""Below is a list of properties that are found in LLM outputs. I would like to summarize this list to AT MOST {num_reduced_axes} representative properties with concise descriptions. Are there any overarching properties that are present in a large number of the properties?

Here is the list of properties:
{{properties}}

Your final list of simplified properties should be human interpretable. The final list of descriptions should be unambiguous and concise. For example, 
* "uses a lot of emojis and markdown" is not a good property because a piece of text can have emojies but not markdown, and vice versa. This should be split into two properties: "uses a lot of emojis" and "uses markdown".
* if two properties are "uses markdown" and "utilizes extensive formatting", text which contains one likely contains the other and should be combined into a single property "uses extensive markdown formatting".
* "focus on historical context" is not a good property because it is too vague. A better property would be "mentions specific historical events".

Each property should be <= 10 words. Order your final list of properties by how much they are seen in the data. Your response should be a list deliniated with "-"
"""

def parse_bullets(text: str):
    lines = text.split("\n")
    bullets = []
    for line in lines:
        if line.strip().startswith("-") or line.strip().startswith("*"):
            bullets.append(line.strip().lstrip("- *").strip())
        else:
            bullets.append(line.strip())
    return bullets

def proposer_postprocess(text: str):
    """
    Process the output from the proposer.
    """
    bullets = parse_bullets(text)
    bullets = [b.replace("**", "").replace("-", "") for b in bullets]
    return bullets


def parse_axes(text: str):
    """
    Parse axes from text.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    axes = []
    for line in lines:
        cleaned = line.strip('1234567890. -"')
        cleaned = cleaned.replace("**", "")
        if cleaned:
            axes.append(cleaned)
    return axes


def propose_vibes(
    df: pd.DataFrame,
    model: str,
    num_proposal_samples: int = 30,
    num_final_vibes: int = 10,
    batch_size: int = 10,
):
    proposer_prompt_freeform = """
You are a machine learning researcher trying to discover interesting behaviors of a single model given a set of questions and responses.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, list the properties or behaviors of the model that are seen in the model responses. Are there any interesting or unexpected behaviors or properties that the model exhibits? Are there any properties that the model does not exhibit that you would expect? How would someone describe the model's behavior? How does the model's behavior differ from the behavior of other models or the behavior of humans?

{combined_responses}

The format should be a list of properties or behaviors that the model exhibits, separated by '-'. For example,
- "formal tone when responding to STEM questions"
- "friendly tone when responding to non-STEM questions"
- "code that optimizes for runtime"
- "gives multiple opinions from different perspectives when asked to give an opinion"
- "uses curse words" 

Note that this example is not at all exhaustive, but rather just an example of the format. Consider properties on many different axes such as tone, language, structure, content, safety, helpfulness, prompt adherence, and any other axis that you can think of. 
    
Remember that these properties should be human interpretable, concise (<= 15 words), substantive and objective. List as many properties as you can find.
"""
    # Create combined responses to get in LOTUS format
    df["single_combined_response"] = df.apply(
        lambda row: (
            f"User prompt:\n{row['question']}\n\n" f"Model:\n{row[model]}\n\n"
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
        proposer_prompt_freeform, return_raw_outputs=True, suffix="properties"
    )

    proposer_df["properties"] = proposer_df["properties"].apply(proposer_postprocess)
    wandb.log({"Vibe Proposer/proposer_results": wandb.Table(dataframe=proposer_df)})
    results = proposer_df[proposer_df["properties"].apply(lambda x: len(x) > 0)]
    results = results.explode("properties").reset_index(drop=True)

    # Cluster and reduce axes
    # TODO: fix groupby_clusterid for sem_agg
    results = results.sem_index("properties", "properties_index").sem_cluster_by(
        "properties", 1
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
    vibin_questions = filtered_vibe_df.sort_values(
        ["vibe", "score"], ascending=[True, False]
    )

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
            axis=1,
        ).tolist()
        input_text = "\n-------------\n".join(input_texts)
        new_df.append({"vibe": vibe, "input_text": input_text})

    new_df = pd.DataFrame(new_df)
    vibe_question_types = new_df.sem_map(
        prompt, return_raw_outputs=True, suffix="vibe_question_types"
    )
    return vibe_question_types


def get_examples_for_vibe(
    vibe_df: pd.DataFrame, vibe: str, model: str, num_examples: int = 5
):
    """Get example pairs where the given vibe was strongly present."""
    vibe_examples = vibe_df[(vibe_df["vibe"] == vibe) & (vibe_df["score"].abs() > 0.0)]
    examples = []
    for _, row in vibe_examples.head(num_examples).iterrows():
        examples.append(
            {
                "prompt": row["question"],
                f"{model}_output": row[model],
                "score": row["score"],
                "core_output": row["raw_outputranker_output_1"],
            }
        )
    return examples


def create_vibe_correlation_plot(vibe_df: pd.DataFrame, model: str):
    """Creates a correlation matrix plot for vibe scores."""
    # Pivot the dataframe to get vibe scores in columns
    vibe_pivot = vibe_df.pivot_table(
        index=["question", model], columns="vibe", values="score"
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


def create_gradio_interface(
    vibe_df: pd.DataFrame,
    model: str,
    output_dir: str,
    model_vibe_scores_plot: go.Figure,
    score_dist_plot: go.Figure,
    vibe_question_types: pd.DataFrame,
):
    """Creates a Gradio interface for visualizing VibeCheck results."""
    import gradio as gr

    def show_examples(selected_vibe):
        examples = get_examples_for_vibe(vibe_df, selected_vibe, model, num_examples=5)
        markdown = f"### What sort of prompts elicit the vibe?\n"
        markdown += f"{vibe_question_types[vibe_question_types['vibe'] == selected_vibe]['vibe_question_types'].values[0]}\n\n"
        markdown += "---\n\n"
        for i, ex in enumerate(examples, 1):
            markdown += f"### Example {i} ({selected_vibe})\n"
            markdown += f"**Prompt:** {ex['prompt']}\n\n"
            markdown += f"**Response:**\n{ex[f'{model}_output']}\n\n"
            markdown += f"**Score:** {ex['score']}\n\n"
            markdown += f"**Judge Output:**\n{ex['core_output']}\n\n"
            markdown += "---\n\n"
        return markdown

    with gr.Blocks() as app:
        gr.Markdown("# VibeCheck Analysis Results")

        with gr.Row():
            with gr.Column():
                gr.Plot(model_vibe_scores_plot)
            with gr.Column():
                gr.Plot(score_dist_plot)

        gr.Markdown("## Example Responses")
        vibe_df_w_types = vibe_df.merge(vibe_question_types, on="vibe", how="left")
        vibe_dropdown = gr.Dropdown(
            choices=sorted(vibe_df_w_types["vibe"].unique()),
            label="Select a vibe to see examples",
        )
        examples_md = gr.Markdown()
        vibe_dropdown.change(show_examples, vibe_dropdown, examples_md)

    return app


def main(
    data_path: str,
    model: str,
    num_proposal_samples: int = 30,
    num_final_vibes: int = 10,
    test: bool = False,
    project_name: str = "vibecheck-single-model",
    proposer_only: bool = False,
    gradio: bool = False,
):
    """Run VibeCheck analysis to identify and analyze behavioral differences between two language models.

    Args:
        data_path (str): Path to CSV file containing model outputs. Must include columns for model responses
            and a 'preference' column indicating which model output was preferred.
        model (str): Name of the model to analyze. This should match the column name in the CSV.
        num_proposal_samples (int, optional): Number of samples to use when proposing vibes. Defaults to 30.
        num_final_vibes (int, optional): Maximum number of vibes to use in final analysis. Defaults to 10.
        test (bool, optional): If True, runs analysis on a small subset of data for testing. Defaults to False.
        project_name (str, optional): Name of the Weights & Biases project. Defaults to "vibecheck".
        proposer_only (bool, optional): If True, only runs the vibe proposal step without analysis. Defaults to False.
        gradio (bool, optional): If True, launches a Gradio interface after analysis. Defaults to False.

    Raises:
        ValueError: If 'preference' column is not found in the input CSV file.
    """
    # Initialize wandb
    wandb.init(project=project_name, name=f"{model}", save_code=True)
    output_dir = f"outputs/{data_path.split('/')[-1].replace('.csv', '')}_{model}"
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

    # df["preference"] = df["preference"].apply(lambda x: 1 if x == model else 0)

    print(f"Preference Counts: {df['preference'].value_counts().to_dict()}")
    wandb.summary["preference_counts"] = df["preference"].value_counts().to_dict()
    wandb.summary["data_size"] = len(df)

    # Create bar plot of preference distribution
    pref_dist = df["preference"].value_counts()
    preference_distribution_plot = go.Figure(
        data=[go.Bar(x=pref_dist.index, y=pref_dist.values, marker_color="#2ecc71")]
    )
    preference_distribution_plot.update_layout(
        title="Model Preference Distribution",
        xaxis_title="Model",
        yaxis_title="Count",
        template="plotly_white",
    )
    wandb.log(
        {"preference_distribution": wandb.Html(preference_distribution_plot.to_html())}
    )
    preference_distribution_plot.write_html(
        os.path.join(output_dir, "preference_distribution.html")
    )
    vibes = propose_vibes(
        df,
        model,
        num_proposal_samples=num_proposal_samples,
        num_final_vibes=num_final_vibes,
    )
    print("Vibes:\n" + "\n".join(vibes))

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
        vibe_df, score_dist_plot = rank_axes(vibes[:3], df, model)
    else:
        vibe_df, score_dist_plot = rank_axes(vibes, df, model)

    # Compute preference alignment
    vibe_df["pref_score"] = vibe_df["score"] * vibe_df["preference"].apply(
        lambda x: 1 if x == model else 0
    )
    vibe_question_types = get_vibe_question_types(vibe_df)
    wandb.log(
        {"Vibe Scoring/vibe_question_types": wandb.Table(dataframe=vibe_question_types)}
    )
    vibe_question_types.to_csv(
        os.path.join(output_dir, "vibe_question_types.csv"), index=False
    )

    wandb.log({"Vibe Scoring/ranker_results": wandb.Table(dataframe=vibe_df)})
    wandb.log({"Vibe Scoring/score_dist_plot": wandb.Html(score_dist_plot.to_html())})
    score_dist_plot.write_html(os.path.join(output_dir, "score_dist_plot.html"))
    vibe_df.to_csv(os.path.join(output_dir, "vibe_df.csv"), index=False)

    agg_df = (
        vibe_df.groupby("vibe")
        .agg({"pref_score": "mean", "score": "mean"})
        .reset_index()
    )
    wandb.log({"summary": wandb.Table(dataframe=agg_df)})

    # plot average scores per vibe
    model_vibe_scores_plot = go.Figure()
    model_vibe_scores_plot.add_trace(
        go.Bar(
            y=agg_df["vibe"],  # vibes on y-axis
            x=agg_df["score"],  # average scores on x-axis
            orientation="h",  # horizontal bars
            text=agg_df["score"].round(2),  # show score values on bars
            textposition="auto",
        )
    )
    model_vibe_scores_plot.update_layout(
        title="Average Vibe Scores",
        xaxis_title="Average Score",
        yaxis_title="Vibe",
        height=max(400, len(agg_df) * 30),  # dynamic height based on number of vibes
        margin=dict(l=200),  # add left margin for vibe labels
    )
    wandb.log(
        {
            "Vibe Plots/model_vibe_scores_plot": wandb.Html(
                model_vibe_scores_plot.to_html()
            )
        }
    )
    model_vibe_scores_plot.write_html(
        os.path.join(output_dir, "model_vibe_scores_plot.html")
    )
    # Filter out vibes with low separation or preference
    vibe_df = vibe_df[vibe_df["score"].abs() > 0.05]
    vibe_df = vibe_df[vibe_df["pref_score"].abs() > 0.05]
    print(
        f"Retained {len(vibe_df.drop_duplicates('vibe'))} vibes with non-trivial separation/preference."
    )
    print("Remaining vibes:\n" + "\n".join(vibe_df["vibe"].unique()))

    # After creating vibe_df and before training models, add:
    corr_plot = create_vibe_correlation_plot(vibe_df, model)
    wandb.log({"Vibe Scoring/vibe_correlations": wandb.Html(corr_plot.to_html())})
    corr_plot.write_html(os.path.join(output_dir, "vibe_correlations.html"))

    # Close wandb run
    wandb.finish()

    # Launch Gradio interface if requested
    if gradio:
        app = create_gradio_interface(
            vibe_df,
            model,
            output_dir,
            model_vibe_scores_plot,
            score_dist_plot,
            vibe_question_types,
        )
        app.launch(share=True)

    return {
        "output_dir": output_dir,
        "model_vibe_scores_plot": model_vibe_scores_plot,
        "score_dist_plot": score_dist_plot,
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
        "--model",
        type=str,
        required=True,
        help="Model to analyze",
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
        "--num_final_vibes",
        type=int,
        default=3,
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

    args = parser.parse_args()
    main(
        args.data_path,
        args.model,
        args.num_proposal_samples,
        args.num_final_vibes,
        args.test,
        args.project,
        args.proposer_only,
        args.gradio,
    )
