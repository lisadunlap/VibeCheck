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
    parse_bullets,
    proposer_postprocess,
    create_reduce_prompt,
    parse_axes,
    get_pref_score,
    # rank_axes,
    parse_vibe_description,
    train_and_evaluate_model,
    get_feature_df,
    ranker_postprocess,
)

def rank_axes(vibes, df, models, position_matters=False):
    """
    Ranks the two model outputs across the given vibes (axes) using LOTUS ranking prompts.
    """

    judge_systems_prompt = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a given property. Which response better aligns more with the given property, A, B, or equal?
When comparing the outputs, consider the following:

- Your sole focus is to determine which response better aligns with the given property, NOT how good or bad the response is.
- Avoid any position bias and remain as objective as possible.
- Consider what the property means and how it applies to the outputs. Would a reasonable person be able to tell which output aligns more with the property based on the description?

Instructions:
- If Response A aligns with the property more than Response B, respond with “A”.
- If Response B aligns with the property more than Response A, respond with “B”.
- If the responses are roughly equal on the property, respond with “equal”.
- If the property does not apply to these outputs (e.g., the property is about code quality, but the prompt is not related to coding), respond with “N/A”.
- If you are unsure about the meaning of the property, respond with “unsure”. 
Think about whether a reasonable person would find the property easy to understand.

A group of humans should agree with your decision. 
Use the following format for your response:
Model: {{A, B, equal, N/A, or unsure}}
"""

    ranker_prompt1 = judge_systems_prompt + """
Here is the property and the two responses:
{ranker_inputs}

Remember to be as objective as possible and strictly adhere to the response format.
"""

    ranker_prompt2 = judge_systems_prompt + """
Here is the property and the two responses:
{ranker_inputs_reversed}

Remember to be as objective as possible and strictly adhere to the response format.
"""

    vibe_dfs = []
    for vibe in vibes:
        vibe_df = df.copy()
        vibe_df["vibe"] = vibe
        vibe_dfs.append(vibe_df)

    vibe_df = pd.concat(vibe_dfs).reset_index(drop=True)
 
    # drop any duplicate columns
    vibe_df = vibe_df.loc[:, ~vibe_df.columns.duplicated()]
    vibe_df["ranker_inputs"] = vibe_df.apply(
        lambda row: f"Property: {row['vibe']}\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[models[0]]}\n\nResponse B:\n{row[models[1]]}",
        axis=1
    )
    if position_matters:
        vibe_df["ranker_inputs_reversed"] = vibe_df.apply(
            lambda row: f"Property: {row['vibe']}\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[models[1]]}\n\nResponse B:\n{row[models[0]]}",
            axis=1
        )

    print(f"Len of df: {len(vibe_df)} num of unique vibes: {len(vibe_df['vibe'].unique())}")
    ranker_1 = vibe_df.sem_map(ranker_prompt1, return_raw_outputs=True, suffix="ranker_output_1")
    print(f"Len of ranker_1: {len(ranker_1)}, num of unique vibes: {len(ranker_1['vibe'].unique())}")
    vibe_df = vibe_df.merge(ranker_1[["vibe", "question", models[0], models[1], "preference", "ranker_output_1", "raw_outputranker_output_1"]], on=["vibe", "question", models[0], models[1], "preference"], how="left")
    vibe_df["ranker_output_1"] = vibe_df["ranker_output_1"].apply(ranker_postprocess)
    print(vibe_df.columns)
    wandb.log({"ranker_1_results": wandb.Table(dataframe=vibe_df[["question", models[0], models[1], "vibe", "preference", "ranker_output_1", "raw_outputranker_output_1"]])})
    print(vibe_df.groupby("vibe")[["ranker_output_1"]].mean())
    if position_matters:
        ranker_2 = vibe_df.sem_map(ranker_prompt2, return_raw_outputs=True, suffix="ranker_output_2")
        vibe_df = vibe_df.merge(ranker_2[["question", models[0], models[1], "preference", "ranker_output_2"]], on=["question", models[0], models[1], "preference"], how="left")
        vibe_df["ranker_output_2"] = vibe_df["ranker_output_2"].apply(ranker_postprocess)
        # Detect if position matters
        vibe_df["position_matters"] = vibe_df["ranker_output_1"] != -1 * vibe_df["ranker_output_2"]
        vibe_df["score"] = vibe_df.apply(
            lambda row: row["ranker_output_1"] if not row["position_matters"] else 0, 
            axis=1
        )
    else:
        vibe_df["score"] = vibe_df["ranker_output_1"]

    return vibe_df

def main(data_path, models, num_proposal_samples=30, num_final_vibes=10, test=False):
    # Initialize wandb
    wandb.init(project="vibecheck", name="model_analysis")

    # Initialize LOTUS
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
    wandb.log({"preference_distribution": wandb.Table(dataframe=df['preference'].value_counts().reset_index())})

    # Create combined responses
    df["combined_responses"] = df.apply(
        lambda row: (
            f"User prompt:\n{row['question']}\n\n"
            f"Model 1:\n{row[models[0]]}\n\n"
            f"Model 2:\n{row[models[1]]}"
        ),
        axis=1
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
    proposer_df = proposer_df.sem_map(proposer_prompt_freeform, return_raw_outputs=True, suffix="differences")
    proposer_df["differences"] = proposer_df["differences"].apply(proposer_postprocess)
    results = proposer_df[proposer_df["differences"].apply(lambda x: len(x) > 0)]
    results = results.explode("differences").reset_index(drop=True)

    # Cluster and reduce axes
    results = results.sem_index("differences", "differences_index").sem_cluster_by("differences", 1)
    # TODO: create PR in LOTUS repo to fix this
    summaries = results.sem_agg(create_reduce_prompt(num_final_vibes), group_by="cluster_id", suffix="reduced axes")
    # summaries = results.sem_agg(create_reduce_prompt(num_final_vibes), suffix="reduced axes")
    summaries["reduced axes parsed"] = summaries["reduced axes"].apply(parse_axes)
    vibes = summaries.explode("reduced axes parsed")["reduced axes parsed"].to_list()
    print("Vibes:\n" + "\n".join(vibes))

    # Log vibes to wandb
    vibes_df = pd.DataFrame({"vibes": vibes})
    wandb.log({"vibes": wandb.Table(dataframe=vibes_df)})
    vibes_df.to_csv("vibes.csv", index=False)

    # Rank axes
    # Change to gpt-4o-mini
    lm = LM(model="gpt-4o-mini", cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=False)
    if test:
        vibe_df = rank_axes(vibes[:3], df, models)
    else:
        vibe_df = rank_axes(vibes, df, models)

    wandb.log({"ranker_results": wandb.Table(dataframe=vibe_df)})
    vibe_df.to_csv("vibe_df.csv", index=False)
    print(vibe_df.columns)

    # Compute preference alignment
    vibe_df["preference_feature"] = vibe_df["preference"].apply(lambda x: get_pref_score(x, models))
    vibe_df["pref_score"] = vibe_df["score"] * vibe_df["preference_feature"]

    agg_df = vibe_df.groupby("vibe").agg({"pref_score": "mean", "score": "mean"}).reset_index()
    wandb.log({"summary": wandb.Table(dataframe=agg_df)})

    # Get the aggregated data and parse descriptions
    agg_df = vibe_df.groupby("vibe").agg({
        "pref_score": "mean", 
        "score": "mean"
    }).reset_index()

    # Split vibe column into name and descriptions
    desc_df = agg_df['vibe'].apply(parse_vibe_description)
    agg_df = pd.concat([agg_df, desc_df], axis=1)

        # Create subplots for side-by-side visualization
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=('Preference Prediction', 'Model Identity'),
        horizontal_spacing=0.1,
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Bar for preference coefficients (left plot)
    fig.add_trace(
        go.Bar(
            y=agg_df['name'],
            x=agg_df['pref_score'],
            name='Preference Prediction',
            orientation='h',
            marker_color='#3498db',
            hovertemplate=(
                '%{x:.4f}<br>' +
                agg_df.apply(
                    lambda row: row['high_desc'] if row['pref_score'] >= 0 else row['low_desc'], axis=1
                ) +
                '<extra></extra>'
            )
        ),
        row=1, col=1
    )

    # Bar for model identity coefficients (right plot)
    fig.add_trace(
        go.Bar(
            y=agg_df['name'],
            x=agg_df['score'],
            name='Model Identity',
            orientation='h',
            marker_color='#2ecc71',
            hovertemplate=(
                '%{x:.4f}<br>' +
                agg_df.apply(
                    lambda row: row['high_desc'] if row['score'] >= 0 else row['low_desc'], axis=1
                ) +
                '<extra></extra>'
            )
        ),
        row=1, col=2
    )

    fig.update_layout(
        title={
            'text': 'Feature Importance Heuristics<br><sup>Mouse over bars to see descriptions of high/low values</sup>',
            'xanchor': 'center',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 20}
        },
        template='plotly_white',
        showlegend=True,
    )

    # Update x-axes
    fig.update_xaxes(
        title_text=f'Seperability Score (Identity {models[0]} vs {models[1]})',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text=f'Seperability Score (Preference {models[0]} vs {models[1]})',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        row=1, col=2
    )

    # Update y-axes (hide right plot's y-axis)
    fig.update_yaxes(
        title_text='',
        row=1,
        col=1,
        ticksuffix='   '  # Add space after tick labels
    )
    fig.update_yaxes(title_text='', showticklabels=False, row=1, col=2)

    wandb.log({"model_vibe_scores_plot": wandb.Html(fig.to_html())})

    # Filter out vibes with low separation or preference
    vibe_df = vibe_df[vibe_df["score"].abs() > 0.05]
    vibe_df = vibe_df[vibe_df["pref_score"].abs() > 0.05]
    print(f"Retained {len(vibe_df.drop_duplicates('vibe'))} vibes with non-trivial separation/preference.")
    print("Remaining vibes:\n" + "\n".join(vibe_df["vibe"].unique()))

    # Prepare training/test data for logistic regression
    feature_df, X_pref, y_pref, y_identity = get_feature_df(vibe_df)
    # feature_df_test, X_pref_test, y_pref_test, y_identity_test = get_feature_df(vibe_df, split="test")

    # Train logistic regressions
    preference_model, preference_coef_df, preference_accuracy_test = train_and_evaluate_model(
        X_pref, y_pref, feature_df.columns, "Preference Prediction"
    )
    identity_model, identity_coef_df, identity_accuracy_test = train_and_evaluate_model(
        X_pref, y_identity, feature_df.columns, "Model Identity Classification"
    )

    wandb.log({
        "preference_model_test_accuracy": preference_accuracy_test,
        "identity_model_test_accuracy": identity_accuracy_test
    })

    # Merge coefficient data
    coef_df = identity_coef_df.merge(
        preference_coef_df, on="vibe", suffixes=("_modelID", "_preference")
    ).sort_values("coef_preference", ascending=False)

    # Parse vibe descriptions for visualization
    desc_df = coef_df['vibe'].apply(parse_vibe_description)
    coef_df = pd.concat([coef_df, desc_df], axis=1)

    # Create subplots for side-by-side visualization
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=('Preference Prediction', 'Model Identity'),
        horizontal_spacing=0.1,
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Bar for preference coefficients (left plot)
    fig.add_trace(
        go.Bar(
            y=coef_df['name'],
            x=coef_df['coef_preference'],
            name='Preference Prediction',
            orientation='h',
            marker_color='#3498db',
            hovertemplate=(
                '%{x:.4f}<br>' +
                coef_df.apply(
                    lambda row: row['high_desc'] if row['coef_preference'] >= 0 else row['low_desc'], axis=1
                ) +
                '<extra></extra>'
            )
        ),
        row=1, col=1
    )

    # Bar for model identity coefficients (right plot)
    fig.add_trace(
        go.Bar(
            y=coef_df['name'],
            x=coef_df['coef_modelID'],
            name='Model Identity',
            orientation='h',
            marker_color='#2ecc71',
            hovertemplate=(
                '%{x:.4f}<br>' +
                coef_df.apply(
                    lambda row: row['high_desc'] if row['coef_modelID'] >= 0 else row['low_desc'], axis=1
                ) +
                '<extra></extra>'
            )
        ),
        row=1, col=2
    )

    fig.update_layout(
        title={
            'text': 'Feature Importance Coefficients<br><sup>Mouse over bars to see descriptions of high/low values</sup>',
            'xanchor': 'center',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 20}
        },
        template='plotly_white',
        showlegend=True,
    )

    # Update x-axes
    fig.update_xaxes(
        title_text='Coefficient Value',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text='Coefficient Value',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        row=1, col=2
    )

    # Update y-axes (hide right plot's y-axis)
    fig.update_yaxes(
        title_text='',
        row=1,
        col=1,
        ticksuffix='   '  # Add space after tick labels
    )
    fig.update_yaxes(title_text='', showticklabels=False, row=1, col=2)

    wandb.log({"model_vibe_coef_plot": wandb.Html(fig.to_html())})

    # Save results
    fig.write_html("vibecheck_results.html")
    coef_df.to_csv("vibecheck_coefficients.csv", index=False)

    # Log final data
    wandb.log({
        "coefficient_data": wandb.Table(dataframe=coef_df)
    })

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run VibeCheck analysis on model outputs.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV file containing model outputs')
    parser.add_argument('--models', nargs="+", default=["command_xlarge_beta", "TNLGv2"],
                        help='Models to compare')
    parser.add_argument('--num_proposal_samples', type=int, default=30,
                        help='Number of samples to use for proposing vibes')
    parser.add_argument('--project', type=str, default="vibecheck",
                        help='Wandb project name')
    parser.add_argument('--test', action='store_true', 
                        help='Run in test mode')
    parser.add_argument('--position_matters', action='store_true', 
                        help='Rerun ranker with different positions')
    parser.add_argument('--num_final_vibes', type=int, default=10,
                        help='Number of final vibes to use for analysis')

    args = parser.parse_args()
    main(args.data_path, args.models, args.num_samples, args.num_final_vibes, args.test)