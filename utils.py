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
            X_train = X
            X_test = X
            y_train = y
            y_test = y

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

def parse_bullets(text: str):
    """
    Parse bullet points from text.
    """
    lines = text.split("\n")
    bullets = []
    for line in lines:
        if line.strip().startswith("-") or line.strip().startswith("*"):
            bullets.append(line.strip().lstrip("- *").strip())
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


def ranker_postprocess(output: str) -> int:
    """
    Postprocess the ranker's output to extract whether model A is favored (1), B is favored (-1), or tie/NA (0).
    """
    try:
        output = output.replace("Output ", "").replace("output ", "")
        output = re.sub(r"[#*]", "", output)
        score_pattern = re.compile(r"Model: (A|B|N/A|unsure|equal)", re.I | re.M)
        score = score_pattern.findall(output)
        if not score:
            return 0
        if score[0].lower() == "a":
            return 1
        elif score[0].lower() == "b":
            return -1
        else:
            return 0
    except Exception as e:
        print(f"Error in ranker_postprocess: {output}")
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
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # Wrap long labels
    def wrap_text(text, width=100):
        if len(text) <= width:
            return text
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        return "<br>".join(lines)

    # Apply wrapping to y-axis labels
    wrapped_labels = [wrap_text(str(label)) for label in df[y_col]]
    
    for i, (x_col, color) in enumerate(zip(x_cols, colors), 1):
        error_x = None
        if error_cols:
            error_x = dict(
                type="data", array=df[error_cols[i - 1]], visible=True, color="#2c3e50"
            )

        fig.add_trace(
            go.Bar(
                y=wrapped_labels,  # Use wrapped labels instead of df[y_col]
                x=df[x_col],
                name=titles[i - 1],
                orientation="h",
                marker_color=color,
                error_x=error_x,
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
