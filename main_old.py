#!/usr/bin/env python3

import argparse
import wandb
import pandas as pd

# Lotus imports
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.cache import CacheConfig, CacheType, CacheFactory

# Plotly
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# Local utility functions
from utils import (
    proposer_postprocess,
    parse_axes,
    get_pref_score,
    parse_vibe_description,
    train_and_evaluate_model,
    get_feature_df,
    ranker_postprocess,
)

def create_reduce_prompt(num_reduced_axes: int):
    return f"""Below is a list of axes with a description of what makes a piece of text low or high on this axis. I would like to summarize this list to at most {num_reduced_axes} representative axes with concise descriptions.

Here is the list of axes:
{{differences}}

These axes should contain only one concept and should be human interpretable. The axis title should be a single concept (does not contain "and", "or", etc.). The descriptions of what makes a piece of text high or low on the axis should be unambiguous and mutually exclusive.

Some examples of BAD axes include:
- "Configuration Clarity: High: Clearly defined structure and purpose. Low: Vaguely defined, minimal purpose." - unclear what the axis is about
- "Language and Communication: High: Varied/precise, complex structure. Low: Straightforward, simple or general language." - describes two separate axes, description is not mutually exclusive. Axis title contains "and", indicating multiple axes.
- "Content Quality: High: High quality, engaging, informative. Low: Low quality, unengaging, uninformative." - this describes multiple axes and "quality" is not well defined

Some examples of GOOD axes include:
- "Formality: High: Informal language. Low: Formal language."
- "Tone: High: Sarcastic tone. Low: Serious tone."
- "Efficiency (coding): High: Optimized for runtime and memory. Low: Brute force algorithms with high memory usage."

Make sure the high and low descriptions are as concise as possible. Please return the simplified list of <={num_reduced_axes} axes with any similar, unclear, or uncommon axes removed. Remember that there may be <{num_reduced_axes} axes which are unique, so double check that you have not returned any simplified axes which are very similar to each other.

Please maintain the format of the original axes and return a numbered list. Each element should be structured as follows:
"{{{{axis_name}}}}: High: {{{{high description}}}} Low: {{{{low description}}}}" 
"""

def rank_axes(vibes, df, models, position_matters=False):
    """
    Ranks the two model outputs across the given vibes (axes) using LOTUS ranking prompts.
    """
    judge_systems_prompt = """You are a fair and unbiased judge comparing two language model responses (A and B) on a given property. Your task is to determine which response better aligns with the HIGH definition of the property.

Instructions:
- Focus solely on which response ranks higher on the property based on its definition
- Read the high/low descriptions carefully and remain objective
- Respond with:
  "A" if Response A better aligns with the "HIGH:" definition compared to Response B
  "B" if Response B better aligns with the "HIGH:" definition compared to Response A
  "equal" if responses are roughly equal
  "N/A" if the property doesn't apply
  "unsure" if the property is poorly defined
"""

    ranker_prompt1 = (
        judge_systems_prompt
        + """
Here is the property and the two responses:
{ranker_inputs}

Remember to output the response that better aligns with the "HIGH:" definition of the property in the format:
Explanation: {{your explanation}}
Model: {{A, B, equal, N/A, or unsure}}
"""
    )

    ranker_prompt2 = (
        judge_systems_prompt
        + """
Here is the property and the two responses:
{ranker_inputs_reversed}

Remember to output the response that better aligns with the "HIGH:" definition of the property in the format:
Explanation: {{your explanation}}
Model: {{A, B, equal, N/A, or unsure}}
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
    if position_matters:
        vibe_df["ranker_inputs_reversed"] = vibe_df.apply(
            lambda row: f"Property: {row['vibe']}\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[models[1]]}\n\nResponse B:\n{row[models[0]]}\n\nProperty (restated): {row['vibe']}",
            axis=1,
        )

    ranker_1 = vibe_df.sem_map(
        ranker_prompt1, return_raw_outputs=True, suffix="ranker_output_1"
    )
    print(
        f"Len of ranker_1: {len(ranker_1)}, num of unique vibes: {len(ranker_1['vibe'].unique())}"
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
    if position_matters:
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
        wandb.summary["position_matters"] = vibe_df["position_matters"].mean()
    else:
        vibe_df["score"] = vibe_df["ranker_output_1"]

    return vibe_df


def main(
    data_path,
    models,
    num_proposal_samples=30,
    num_final_vibes=10,
    test=False,
    position_matters=False,
    project_name="vibecheck",
):
    # Initialize wandb
    wandb.init(project=project_name, name=f"{models[0]}_vs_{models[1]}", save_code=True)

    # Initialize LOTUS
    # TODO: create PR in LOTUS repo to fix cahcing problems
    cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)
    lm = LM(model="gpt-4o", cache=cache)
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    lotus.settings.configure(lm=lm, rm=rm, enable_cache=False)

    # Load and preprocess data
    df = pd.read_csv(data_path)
    if test:
        df = df.sample(100, random_state=42)
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

    # Create combined responses
    df["combined_responses"] = df.apply(
        lambda row: (
            f"User prompt:\n{row['question']}\n\n"
            f"Model 1:\n{row[models[0]]}\n\n"
            f"Model 2:\n{row[models[1]]}"
        ),
        axis=1,
    )

    # Propose vibes
    proposer_prompt_freeform = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. Write down as many differences as you can find between the two outputs. Please format your differences as a list of axes of variation and differences between the two outputs. Try to give axes which represent a property that a human could easily interpret and they could categorize a pair of text outputs as higher or lower on that specific axis.
a pair of text outputs as higher or lower on that specific axis.

Here is the question and the two responses:
{combined_responses}

The format should be:
- {{axis_1}}: {{difference}}
- {{axis_2}}: {{difference}}
    
Remember that these axes should be human interpretable and that the differences should be substantive and objective. 
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

    # Run on a subsample for the "train" split
    proposer_df = df.sample(num_proposal_samples, random_state=42)
    proposer_df = proposer_df.sem_map(
        proposer_prompt_freeform, return_raw_outputs=True, suffix="differences"
    )
    proposer_df["differences"] = proposer_df["differences"].apply(proposer_postprocess)
    results = proposer_df[proposer_df["differences"].apply(lambda x: len(x) > 0)]
    results = results.explode("differences").reset_index(drop=True)

    # Cluster and reduce axes
    results = results.sem_index("differences", "differences_index").sem_cluster_by(
        "differences", 1
    )
    summaries = results.sem_agg(
        create_reduce_prompt(num_final_vibes),
        group_by="cluster_id",
        suffix="reduced axes",
    )
    # summaries = results.sem_agg(create_reduce_prompt(num_final_vibes), suffix="reduced axes")
    summaries["reduced axes parsed"] = summaries["reduced axes"].apply(parse_axes)
    vibes = summaries.explode("reduced axes parsed")["reduced axes parsed"].to_list()
    print("Vibes:\n" + "\n".join(vibes))

    # Log vibes to wandb
    vibes_df = pd.DataFrame({"vibes": vibes})
    wandb.log({"vibes": wandb.Table(dataframe=vibes_df)})
    vibes_df.to_csv("vibes.csv", index=False)

    # Rank axes
    lm = LM(model="gpt-4o-mini", cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=False)
    if test:
        vibe_df = rank_axes(vibes[:3], df, models, position_matters=position_matters)
    else:
        vibe_df = rank_axes(vibes, df, models, position_matters=position_matters)

    wandb.log({"ranker_results": wandb.Table(dataframe=vibe_df)})
    vibe_df.to_csv("vibe_df.csv", index=False)
    print(vibe_df.columns)

    # Compute preference alignment
    vibe_df["preference_feature"] = vibe_df["preference"].apply(
        lambda x: get_pref_score(x, models)
    )
    vibe_df["pref_score"] = vibe_df["score"] * vibe_df["preference_feature"]

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

    # Split vibe column into name and descriptions
    desc_df = agg_df["vibe"].apply(parse_vibe_description)
    agg_df = pd.concat([agg_df, desc_df], axis=1)

    # Create subplots for side-by-side visualization
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Preference Prediction", "Model Identity"),
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # Bar for preference coefficients (left plot)
    fig.add_trace(
        go.Bar(
            y=agg_df["name"],
            x=agg_df["pref_score"],
            name="Preference Prediction",
            orientation="h",
            marker_color="#3498db",
            hovertemplate=(
                "%{x:.4f}<br>"
                + agg_df.apply(
                    lambda row: (
                        row["high_desc"] if row["pref_score"] >= 0 else row["low_desc"]
                    ),
                    axis=1,
                )
                + "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Bar for model identity coefficients (right plot)
    fig.add_trace(
        go.Bar(
            y=agg_df["name"],
            x=agg_df["score"],
            name="Model Identity",
            orientation="h",
            marker_color="#2ecc71",
            hovertemplate=(
                "%{x:.4f}<br>"
                + agg_df.apply(
                    lambda row: (
                        row["high_desc"] if row["score"] >= 0 else row["low_desc"]
                    ),
                    axis=1,
                )
                + "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title={
            "text": "Feature Importance Heuristics<br><sup>Mouse over bars to see descriptions of high/low values</sup>",
            "xanchor": "center",
            "y": 0.95,
            "x": 0.5,
            "font": {"size": 20},
        },
        template="plotly_white",
        showlegend=True,
    )

    # Update x-axes
    fig.update_xaxes(
        title_text=f"Seperability Score (Identity {models[0]} vs {models[1]})",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text=f"Seperability Score (Preference {models[0]} vs {models[1]})",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black",
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="", row=1, col=1, ticksuffix="   ")
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=2)

    wandb.log({"model_vibe_scores_plot": wandb.Html(fig.to_html())})

    # Filter out vibes with low separation or preference
    vibe_df = vibe_df[vibe_df["score"].abs() > 0.05]
    vibe_df = vibe_df[vibe_df["pref_score"].abs() > 0.05]
    print(
        f"Retained {len(vibe_df.drop_duplicates('vibe'))} vibes with non-trivial separation/preference."
    )
    print("Remaining vibes:\n" + "\n".join(vibe_df["vibe"].unique()))

    feature_df, X_pref, y_pref, y_identity = get_feature_df(vibe_df)
    (
        preference_model,
        preference_coef_df,
        preference_accuracy_test,
        preference_acc_std,
    ) = train_and_evaluate_model(
        X_pref, y_pref, feature_df.columns, "Preference Prediction"
    )
    identity_model, identity_coef_df, identity_accuracy_test, identity_acc_std = (
        train_and_evaluate_model(
            X_pref, y_identity, feature_df.columns, "Model Identity Classification"
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

    # Parse vibe descriptions for visualization
    desc_df = coef_df["vibe"].apply(parse_vibe_description)
    coef_df = pd.concat([coef_df, desc_df], axis=1)

    # Create subplots for side-by-side visualization
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Preference Prediction", "Model Identity"),
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # Bar for preference coefficients (left plot)
    fig.add_trace(
        go.Bar(
            y=coef_df["name"],
            x=coef_df["coef_preference"],
            name="Preference Prediction",
            orientation="h",
            marker_color="#3498db",
            error_x=dict(
                type="data",
                array=coef_df["coef_std_preference"],
                visible=True,
                color="#2c3e50",
            ),
            hovertemplate=(
                "%{x:.4f} ± %{error_x:.4f}<br>"
                + coef_df.apply(
                    lambda row: (
                        row["high_desc"]
                        if row["coef_preference"] >= 0
                        else row["low_desc"]
                    ),
                    axis=1,
                )
                + "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Bar for model identity coefficients (right plot)
    fig.add_trace(
        go.Bar(
            y=coef_df["name"],
            x=coef_df["coef_modelID"],
            name="Model Identity",
            orientation="h",
            marker_color="#2ecc71",
            error_x=dict(
                type="data",
                array=coef_df["coef_std_modelID"],
                visible=True,
                color="#2c3e50",
            ),
            hovertemplate=(
                "%{x:.4f} ± %{error_x:.4f}<br>"
                + coef_df.apply(
                    lambda row: (
                        row["high_desc"]
                        if row["coef_modelID"] >= 0
                        else row["low_desc"]
                    ),
                    axis=1,
                )
                + "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title={
            "text": "Feature Importance Coefficients<br><sup>Mouse over bars to see descriptions of high/low values</sup>",
            "xanchor": "center",
            "y": 0.95,
            "x": 0.5,
            "font": {"size": 20},
        },
        template="plotly_white",
        showlegend=True,
    )

    # Update x-axes
    fig.update_xaxes(
        title_text="Coefficient Value",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Coefficient Value",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black",
        row=1,
        col=2,
    )

    # Update y-axes (hide right plot's y-axis)
    fig.update_yaxes(
        title_text="", row=1, col=1, ticksuffix="   "  # Add space after tick labels
    )
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=2)

    wandb.log({"model_vibe_coef_plot": wandb.Html(fig.to_html())})

    # Save results
    fig.write_html("vibecheck_results.html")
    coef_df.to_csv("vibecheck_coefficients.csv", index=False)

    # Log final data
    wandb.log({"coefficient_data": wandb.Table(dataframe=coef_df)})

    # Close wandb run
    wandb.finish()


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
        default=["command_xlarge_beta", "TNLGv2"],
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
        "--position_matters",
        action="store_true",
        help="Rerun ranker with different positions",
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

    args = parser.parse_args()
    main(
        args.data_path,
        args.models,
        args.num_proposal_samples,
        args.num_final_vibes,
        args.test,
        args.position_matters,
    )
