#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from sklearn.utils import shuffle


def train_and_evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    split_train_test: bool = True,
    solver: str = "elasticnet",
    n_splits: int = 5,
):
    """
    Train a logistic regression model on (X, y) and compute accuracy and p-values for each feature.

    Args:
        X: Feature matrix
        y: Target values
        feature_names: Names of features
        split_train_test: Whether to split data into train/test sets
        solver: Type of regularization to use ('standard', 'lasso', or 'elasticnet')
        bootstrap_iters: Number of bootstrap iterations (0 for no bootstrapping)
        n_splits: Number of random train/test splits to average over
    """
    # Configure model based on solver type
    if solver == "standard":
        model = LogisticRegression(penalty="l2", random_state=42)
    elif solver == "lasso":
        model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
    elif solver == "elasticnet":
        model = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, random_state=42
        )
    else:
        raise ValueError("solver must be one of: 'standard', 'lasso', 'elasticnet'")

    # Initialize arrays to store results across splits
    split_accuracies = []
    split_coefs = []

    for split in range(n_splits):
        # Split and train
        if split_train_test:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=42 + split
            )
        else:
            print("Using all data for training and testing")
            X_train, y_train = shuffle(X, y, random_state=42 + split)
            X_test, y_test = X, y

        model.fit(X_train, y_train)

        # Store results for this split
        split_accuracies.append(accuracy_score(y_test, model.predict(X_test)))
        split_coefs.append(model.coef_[0])

    # Average results across splits
    accuracy = np.mean(split_accuracies)
    model.coef_ = np.mean(split_coefs, axis=0).reshape(1, -1)

    if n_splits > 1:
        acc_std = np.std(split_accuracies)
        coef_std = np.std(split_coefs, axis=0)
        print(f"Accuracy ({solver}): {accuracy:.3f} ± {acc_std:.3f}")
        print(f"Coefficients ({solver}): {model.coef_[0]} ± {coef_std}")
    else:
        acc_std = 0
        print(f"Accuracy ({solver}): {accuracy:.3f}")

    # Calculate p-values
    X_with_intercept = np.column_stack([np.ones(len(X_train)), X_train])
    predictions = model.predict(X_train)
    mse = np.sum((predictions - y_train) ** 2) / (
        len(y_train) - X_with_intercept.shape[1]
    )
    var_covar_matrix = mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
    standard_errors = np.sqrt(np.diag(var_covar_matrix))[1:]
    z_scores = model.coef_[0] / standard_errors
    p_values = 2 * (1 - stats.norm.cdf(abs(z_scores)))

    # Create dataframe with feature names, coefficients, and p-values
    coef_df = pd.DataFrame(
        {
            "vibe": feature_names,
            "coef": model.coef_[0],
            "p_value": p_values,
        }
    )

    # If using multiple splits, add split-based confidence intervals
    if n_splits > 1:
        coef_df["coef_std"] = np.std(split_coefs, axis=0)
        coef_df["coef_lower_split"] = model.coef_[0] - 1.96 * coef_df["coef_std"]
        coef_df["coef_upper_split"] = model.coef_[0] + 1.96 * coef_df["coef_std"]

    return model, coef_df, accuracy, acc_std


def get_feature_df(vibe_df: pd.DataFrame):
    """
    Given a vibe_df with "score" columns pivoted by "vibe", construct X, y
    arrays for preference and identity classification.
    """
    # Pivot to create wide-format scores for each vibe
    feature_df = pd.pivot_table(
        vibe_df, values="score", index=vibe_df.index, columns="vibe", fill_value=0
    )
    print(feature_df.columns)

    feature_df_1 = feature_df.copy()
    feature_df_2 = -1 * feature_df.copy()

    # Preference data
    X_pref = np.vstack([feature_df_1.to_numpy(), feature_df_2.to_numpy()])
    y_pref = np.concatenate(
        [
            vibe_df["preference_feature"][: len(feature_df)].to_numpy(),
            -1 * vibe_df["preference_feature"][: len(feature_df)].to_numpy(),
        ]
    )

    # Model identity data
    y_identity = np.concatenate(
        [np.ones(len(feature_df_1)), -1 * np.ones(len(feature_df_2))]
    )

    return feature_df, X_pref, y_pref, y_identity


def get_pref_score(preference: str, models: list):
    """
    Get preference score based on model preference.
    """
    if preference == models[0]:
        return 1
    elif preference == models[1]:
        return -1
    else:
        return 0


def parse_vibe_description(vibe_text: str) -> pd.Series:
    """
    Split an axis string (like "Complexity: High: ... Low: ...") into structured data.
    """
    if "High:" not in vibe_text or "Low:" not in vibe_text:
        return pd.Series({"name": vibe_text, "high_desc": "", "low_desc": ""})

    parts = vibe_text.split("High:")
    name = parts[0].strip(": ")
    high_low_parts = parts[1].split("Low:")
    high_desc = high_low_parts[0].strip()
    low_desc = high_low_parts[1].strip()
    return pd.Series({"name": name, "high_desc": high_desc, "low_desc": low_desc})


def create_side_by_side_plot(
    df,
    y_col,
    x_cols,
    titles,
    main_title,
    models,
    error_cols=None,
    colors=("#2ecc71", "#3498db"),
):
    """Creates a side-by-side horizontal bar plot with two subplots.

    Args:
        df: DataFrame containing the data
        y_col: Column name for y-axis labels
        x_cols: List of two column names for x-axis values
        titles: List of two subplot titles
        main_title: Main title for the entire plot
        models: List of model names for x-axis labels
        error_cols: Optional list of two column names for error bars
        colors: Tuple of two colors for the bars
    """
    df = df.sort_values(by=x_cols[0], ascending=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # Truncate labels to first 5 words but keep original for hover
    def truncate_text(text, num_words=5):
        words = str(text).split()
        if len(words) <= num_words:
            return " ".join(words)
        return " ".join(words[:num_words]) + "..."

    # Create both truncated and full labels
    truncated_labels = [truncate_text(label) for label in df[y_col]]
    full_labels = [str(label) for label in df[y_col]]

    for i, (x_col, color) in enumerate(zip(x_cols, colors), 1):
        error_x = None
        if error_cols:
            error_x = dict(
                type="data", array=df[error_cols[i - 1]], visible=True, color="#2c3e50"
            )

        fig.add_trace(
            go.Bar(
                y=truncated_labels,  # Truncated labels for y-axis
                x=df[x_col],
                name=titles[i - 1],
                orientation="h",
                marker_color=color,
                error_x=error_x,
                hovertemplate="<b>%{customdata}</b><br>"  # Show full label in hover
                + "Value: %{x}<br>"
                + "<extra></extra>",
                customdata=full_labels,  # Full labels for hover
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        title={
            "text": main_title,
            "xanchor": "center",
            "y": 0.95,
            "x": 0.5,
            "font": {"size": 20},
        },
        template="plotly_white",
        showlegend=True,
    )

    for i, subtitle in enumerate(
        [
            f"Seperability Score<br>{models[0]}(+) vs {models[1]}(-)",
            f"Seperability Score<br>Preferred(+) vs Unpreferred(-)",
        ],
        1,
    ):
        fig.update_xaxes(
            title_text=subtitle,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            row=1,
            col=i,
        )

    fig.update_yaxes(title_text="", row=1, col=1, ticksuffix="   ")
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=2)

    return fig


def get_examples_for_vibe(
    vibe_df: pd.DataFrame, vibe: str, models: List[str], num_examples: int = 5
):
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


def create_gradio_app(
    vibe_df: pd.DataFrame,
    models: List[str],
    coef_df: pd.DataFrame,
    corr_plot: go.Figure,
    vibe_question_types: pd.DataFrame,
):
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
    
    vibe_pivot = vibe_df.pivot_table(
        index=["question", models[0], models[1]], columns="vibe", values="score"
    ).reset_index()
    vibe_pivot = vibe_pivot.fillna(0)

    # Calculate correlation matrix for just the vibe scores
    vibe_cols = vibe_pivot.columns[3:]  # Skip the index columns
    corr_matrix = vibe_pivot[vibe_cols].corr()

    # Truncate labels
    def truncate_text(text, num_words=5):
        words = str(text).split()
        if len(words) <= num_words:
            return " ".join(words)
        return " ".join(words[:num_words]) + "..."

    truncated_labels = [truncate_text(col) for col in corr_matrix.columns]
    full_labels = list(corr_matrix.columns)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=truncated_labels,  # Use truncated labels for display
            y=truncated_labels,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate=(
                "x: %{customdata[0]}<br>"  # Show full labels in hover
                "y: %{customdata[1]}<br>"
                "correlation: %{z:.2f}<br>"
                "<extra></extra>"
            ),
            customdata=[[x, y] for x in full_labels for y in full_labels],  # Full labels for hover
        )
    )

    fig.update_layout(
        title="Vibe Score Correlations",
        xaxis_tickangle=-45,
        width=800,
        height=800,
    )

    return fig
