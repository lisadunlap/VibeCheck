#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Tuple
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

def train_and_evaluate_model(
    vibe_df: pd.DataFrame,
    models: List[str],
    label: str,
    split_train_test: bool = True,
    solver: str = "elasticnet",
    n_bootstrap: int = 1000,
):
    """
    Train a logistic regression model using bootstrap resampling to compute accuracy and p-values.

    Args:
        vibe_df: DataFrame containing the vibe scores
        models: List of model names
        label: Target label ('preference' or 'identity')
        split_train_test: Whether to split data into train/test sets
        solver: Type of regularization ('standard', 'lasso', or 'elasticnet')
        n_bootstrap: Number of bootstrap iterations
    """
    feature_df, X, y_pref, y_identity = get_feature_df(vibe_df, models, flip_identity=True)
    if label == "preference":
        y = y_pref
    elif label == "identity":
        y = y_identity
    else:
        raise ValueError("label must be one of: 'preference', 'identity'")
    feature_names = feature_df.columns

    # Normalize all features once before bootstrapping
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Initialize arrays to store bootstrap results
    bootstrap_accuracies = []
    bootstrap_coefs = []
    models = []
    n_samples = len(X)

    for _ in range(n_bootstrap):
        # Create bootstrap sample from normalized data
        rng = np.random.RandomState(None)
        bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[bootstrap_indices]
        y_boot = y[bootstrap_indices]

        # Split bootstrap sample if requested
        if split_train_test:
            X_train, X_test, y_train, y_test = train_test_split(
                X_boot, y_boot, test_size=0.3, random_state=42, stratify=y_boot
            )
        else:
            X_train, y_train = X_boot, y_boot
            X_test, y_test = X, y

        # Create and train model
        if solver == "standard":
            model = LogisticRegression(penalty="l2", random_state=42)
        elif solver == "lasso":
            model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
        elif solver == "elasticnet":
            model = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, random_state=42
            )

        model.fit(X_train, y_train)
        models.append(model)

        # Store results for this bootstrap iteration
        bootstrap_accuracies.append(accuracy_score(y_test, model.predict(X_test)))
        bootstrap_coefs.append(model.coef_[0])

    # Calculate confidence intervals using percentiles
    accuracy = np.mean(bootstrap_accuracies)
    acc_std = np.std(bootstrap_accuracies)
    acc_ci = np.percentile(bootstrap_accuracies, [2.5, 97.5])
    
    # Average coefficients and calculate their confidence intervals
    mean_coefs = np.mean(bootstrap_coefs, axis=0)
    coef_ci = np.percentile(bootstrap_coefs, [2.5, 97.5], axis=0)
    
    # Set the final model coefficients to the bootstrap mean
    model.coef_ = mean_coefs.reshape(1, -1)

    # Get model predictions and correctness
    all_predictions = []
    for m in models:
        predictions = m.predict(X)
        correct = (predictions == y)
        feature_df_copy = feature_df.copy()
        feature_df_copy["correct"] = correct
        all_predictions.append(feature_df_copy)
    all_predictions = pd.concat(all_predictions)
    all_predictions = all_predictions.reset_index()
    avg_correct = all_predictions.groupby("conversation_id")["correct"].mean().to_frame()

    print(f"{label} Accuracy ({solver}): {accuracy:.3f} ± {acc_std:.3f}")
    print(f"{label} 95% CI: [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")

    # Calculate feature importance metrics
    if solver == "standard":
        # Use bootstrap distribution for p-values
        z_scores = mean_coefs / np.std(bootstrap_coefs, axis=0)
        p_values = 2 * (1 - stats.norm.cdf(abs(z_scores)))
    else:
        # For Lasso and Elasticnet, use coefficient stability across bootstrap samples
        coef_nonzero = np.mean([np.abs(c) > 1e-10 for c in bootstrap_coefs], axis=0)
        p_values = 1 - coef_nonzero

    # Create results dataframe
    coef_df = pd.DataFrame({
        "vibe": feature_names,
        "coef": np.mean(bootstrap_coefs, axis=0),
        "selection_frequency": coef_nonzero if solver != "standard" else 1.0,
        "p_value": p_values,
        "coef_std": np.std(bootstrap_coefs, axis=0),
        "coef_lower_ci": coef_ci[0],
        "coef_upper_ci": coef_ci[1]
    })

    # Add stability metric
    coef_df["stability"] = 1 - (coef_df["coef_std"] / np.abs(coef_df["coef"]))
    
    # Sort by appropriate importance metric
    if solver == "standard":
        coef_df = coef_df.sort_values("p_value")
    else:
        coef_df = coef_df.sort_values(
            ["selection_frequency", "stability", "coef_std"],
            ascending=[False, False, True]
        )

    metrics = {
        "accuracy": accuracy,
        "acc_std": acc_std,
        "acc_ci": acc_ci,
    }
    return model, coef_df, avg_correct, metrics


def get_feature_df(vibe_df: pd.DataFrame, models: List[str], flip_identity: bool = False):
    """
    Given a vibe_df with "score" columns pivoted by "vibe", construct X, y
    arrays for preference and identity classification.
    """

    orig_df = vibe_df.drop_duplicates(subset="conversation_id")
    # Pivot to create wide-format scores for each vibe
    feature_df = pd.pivot_table(
        vibe_df, values="score", index="conversation_id", columns="vibe", fill_value=0
    )
    y_pref = orig_df["preference_feature"].to_numpy()
    y_identity = orig_df["score_pos_model"].apply(lambda x: 1 if models[0] == x[0] else -1).to_numpy()

    X_pref = feature_df.to_numpy()

    # if y_identity is all 1, copy X_pref and negate it
    if np.all(y_identity == 1):
        if flip_identity:
            X_pref = np.vstack([X_pref, -1 * X_pref.copy()])
            y_identity = np.concatenate([y_identity, -1 * y_identity])
            y_pref = np.concatenate([y_pref, -1 * y_pref])
            feature_df = pd.concat([feature_df, feature_df.copy()])
        else:
            flipped_indices = np.random.permutation(len(X_pref))[:len(X_pref)//2]
            print(f"Flipping {len(flipped_indices)} rows")
            X_pref[flipped_indices] = -1 * X_pref[flipped_indices]
            y_pref[flipped_indices] = -1 * y_pref[flipped_indices]
            y_identity[flipped_indices] = -1 * y_identity[flipped_indices]

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
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    titles: List[str],
    main_title: str,
    models: List[str],
    error_cols: List[str] = None,
    colors: Tuple[str, str] = ("#2ecc71", "#3498db"),
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

    # Create both truncated and full labels AFTER sorting
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
                customdata=[[label] for label in full_labels],  # Wrap each label in a list
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
        margin=dict(l=20, r=20, t=100, b=20),
    )

    for i, subtitle in enumerate(
        [
            f"Seperability Score<br>{models[0]} is +<br>{models[1]} is -",
            f"Seperability Score<br>Preferred is +<br>Unpreferred is -",
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
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig

class EmbeddingMLP(nn.Module):
    """PyTorch MLP for embedding classification."""
    def __init__(self, input_dim: int, hidden_dims: list = [100, 50]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_embedding_classifier(df: pd.DataFrame, 
                            n_bootstrap: int = 10,
                            device: str = "cuda" if torch.cuda.is_available() else "cpu",
                            batch_size: int = 32,
                            epochs: int = 10,
                            lr: float = 0.001,
                            test_size: float = 0.3) -> dict:
    """
    Train a PyTorch MLP classifier on embedding differences between two models and compute confidence intervals.
    
    Args:
        df: DataFrame containing model_a_embedding and model_b_embedding columns
        n_bootstrap: Number of bootstrap iterations for confidence intervals
        device: Device to run the model on ("cuda" or "cpu")
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        test_size: Proportion of data to use for testing (default: 0.3)
    
    Returns:
        dict: Contains classifier, accuracies, and confidence intervals
    """
    # Create training data
    X_diff_0 = np.array([a - b for a, b in zip(df["model_a_embedding"], df["model_b_embedding"])])
    X_diff_1 = np.array([b - a for a, b in zip(df["model_a_embedding"], df["model_b_embedding"])])
    X = np.vstack([X_diff_0, X_diff_1])
    y = np.hstack([np.zeros(len(df)), np.ones(len(df))])
    y_pref = np.hstack([df["preference_feature"], -1 * df["preference_feature"]])
    # turn -1 into 0
    y_pref = np.where(y_pref == -1, 0, y_pref)
    
    def train_and_bootstrap(X, y, label="model_id"):
        bootstrap_acc = []
        
        for _ in tqdm(range(n_bootstrap), desc=f"Bootstrapping {label}"):
            # Bootstrap sampling
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_boot, y_boot, test_size=test_size, random_state=42
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train).to(device)
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_test_tensor = torch.FloatTensor(y_test).to(device)
            
            # Create data loader
            train_loader = DataLoader(
                TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1)), 
                batch_size=batch_size, 
                shuffle=True
            )
            
            # Initialize model
            model = EmbeddingMLP(input_dim=X.shape[1]).to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluation on test set
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_tensor)
                acc = ((y_pred >= 0.5).float().squeeze() == y_test_tensor).float().mean().item()
            
            bootstrap_acc.append(acc)
        
        mean_acc = np.mean(bootstrap_acc)
        ci = np.percentile(bootstrap_acc, [2.5, 97.5])
        
        return mean_acc, ci
    
    print("Training model_id classifier...")
    model_id_acc, model_id_ci = train_and_bootstrap(X, y, "model_id")
    print(f"Model ID accuracy: {model_id_acc:.3f} (95% CI: [{model_id_ci[0]:.3f}, {model_id_ci[1]:.3f}])")
    
    print("Training preference classifier...")
    preference_acc, preference_ci = train_and_bootstrap(X, y_pref, "preference")
    print(f"Preference accuracy: {preference_acc:.3f} (95% CI: [{preference_ci[0]:.3f}, {preference_ci[1]:.3f}])")

    results = {
        "embedding_mlp_model_id_accuracy": model_id_acc,
        "embedding_mlp_model_id_ci": model_id_ci,
        "embedding_mlp_preference_accuracy": preference_acc,
        "embedding_mlp_preference_ci": preference_ci
    }
    
    return results