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

# Local utility functions
from utils import (
    parse_bullets,
    proposer_postprocess,
    create_reduce_prompt,
    parse_axes,
    get_pref_score,
    rank_axes,
    parse_vibe_description,
    train_and_evaluate_model,
    get_feature_df
)

def main(data_path, num_samples, test=False):
    # Initialize wandb
    wandb.init(project="vibecheck", name="model_analysis")

    # Initialize LOTUS
    cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)
    lm = LM(model="gpt-4o", cache=cache)
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    lotus.settings.configure(lm=lm, rm=rm)

    # Define models to compare
    models = ["command_xlarge_beta", "TNLGv2"]

    # Load and preprocess data
    df = pd.read_csv(data_path)
    if test:
        df = df.sample(100, random_state=42)
    df = df[df["preference"].isin(models)]

    train_df = df.sample(frac=0.5, random_state=42)
    test_df = df[~df.index.isin(train_df.index)]
    train_df['split'] = "train"
    test_df['split'] = "test"
    df = pd.concat([train_df, test_df])

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
    proposer_df = df[df["split"] == "train"].sample(num_samples, random_state=42)
    proposer_df = proposer_df.sem_map(proposer_prompt_freeform, return_raw_outputs=True, suffix="differences")
    proposer_df["differences"] = proposer_df["differences"].apply(proposer_postprocess)
    results = proposer_df[proposer_df["differences"].apply(lambda x: len(x) > 0)]
    results = results.explode("differences").reset_index(drop=True)

    # Cluster and reduce axes
    results = results.sem_index("differences", "differences_index").sem_cluster_by("differences", 1)
    summaries = results.sem_agg(create_reduce_prompt(10), group_by="cluster_id", suffix="reduced axes")
    summaries["reduced axes parsed"] = summaries["reduced axes"].apply(parse_axes)
    vibes = summaries.explode("reduced axes parsed")["reduced axes parsed"].to_list()

    # Log vibes to wandb
    vibes_df = pd.DataFrame({"vibes": vibes})
    wandb.log({"vibes": wandb.Table(dataframe=vibes_df)})
    vibes_df.to_csv("vibes.csv", index=False)

    # Rank axes
    if test:
        vibe_df = rank_axes(vibes[:3], df, models, lm, rm)
    else:
        vibe_df = rank_axes(vibes, df, models, lm, rm)

    wandb.log({"ranker_results": wandb.Table(dataframe=vibe_df)})
    vibe_df.to_csv("vibe_df.csv", index=False)
    print(vibe_df.columns)

    # Compute preference alignment
    vibe_df["preference_feature"] = vibe_df["preference"].apply(lambda x: get_pref_score(x, models))
    vibe_df["pref_score"] = vibe_df["score"] * vibe_df["preference_feature"]

    agg_df = vibe_df.groupby("vibe").agg({"pref_score": "mean", "score": "mean"}).reset_index()
    wandb.log({"summary": wandb.Table(dataframe=agg_df)})

    # Parse vibe descriptions
    desc_df = agg_df['vibe'].apply(parse_vibe_description)
    agg_df = pd.concat([agg_df, desc_df], axis=1)

    # Build the first visualization
    fig = go.Figure()

    # Bar for raw "score"
    fig.add_trace(
        go.Bar(
            y=agg_df['name'],
            x=agg_df['score'],
            name='Score',
            orientation='h',
            marker_color='#3498db',
            hovertemplate=(
                '%{x:.2f}<br>' +
                agg_df.apply(lambda row: row['high_desc'] if row['score'] >= 0 else row['low_desc'], axis=1) +
                '<extra></extra>'
            )
        )
    )

    # Bar for "pref_score"
    fig.add_trace(
        go.Bar(
            y=agg_df['name'],
            x=agg_df['pref_score'],
            name='Preference Score',
            orientation='h',
            marker_color='#2ecc71',
            hovertemplate=(
                '%{x:.2f}<br>' +
                agg_df.apply(lambda row: row['high_desc'] if row['pref_score'] >= 0 else row['low_desc'], axis=1) +
                '<extra></extra>'
            )
        )
    )

    for i, row in agg_df.iterrows():
        fig.add_annotation(
            x=0,
            y=row['name'],
            text=f"<b>{row['name']}</b>",
            showarrow=False,
            yshift=20,
            font=dict(size=14)
        )

    fig.update_layout(
        barmode='group',
        title={
            'text': f'{models[0]} model Vibes<br><sup>Mouse over bars to see the relevant descriptions</sup>',
            'xanchor': 'center',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title='Score',
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(visible=False),
        template='plotly_white',
        showlegend=True
    )

    wandb.log({"feature_importance_plot": wandb.Html(fig.to_html())})

    # Filter out vibes with low separation or preference
    vibe_df = vibe_df[vibe_df["score"].abs() > 0.05]
    vibe_df = vibe_df[vibe_df["pref_score"].abs() > 0.05]
    print(f"Retained {len(vibe_df.drop_duplicates('vibe'))} vibes with non-trivial separation/preference.")
    print("Remaining vibes:\n" + "\n".join(vibe_df["vibe"].unique()))

    # Prepare training/test data for logistic regression
    vibe_df["split"] = "train"
    feature_df, X_pref, y_pref, y_identity = get_feature_df(vibe_df, split="train")
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

    # Create final figure for coefficients
    desc_df = coef_df['vibe'].apply(parse_vibe_description)
    coef_df = pd.concat([coef_df, desc_df], axis=1)

    fig = go.Figure()
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
        )
    )

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
        )
    )

    for name in coef_df['name']:
        fig.add_annotation(x=0, y=name, text=f"<b>{name}</b>", showarrow=False, yshift=20, font=dict(size=14))

    fig.update_layout(
        barmode='group',
        title={
            'text': 'Feature Importance Coefficients<br><sup>Hover to see descriptions of high/low values</sup>',
            'xanchor': 'center',
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title='Coefficient Value',
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(visible=False),
        template='plotly_white',
        showlegend=True,
    )

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
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of samples to use for proposing vibes')
    parser.add_argument('--test', action='store_true', 
                        help='Run in test mode')

    args = parser.parse_args()
    main(args.data_path, args.num_samples, args.test)