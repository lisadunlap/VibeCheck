#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List

def parse_bullets(text: str):
    """
    Parse bullet points from text.
    """
    lines = text.split('\n')
    bullets = []
    for line in lines:
        if line.strip().startswith('-') or line.strip().startswith('*'):
            bullets.append(line.strip().lstrip('- *').strip())
    return bullets

def proposer_postprocess(text: str):
    """
    Process the output from the proposer.
    """
    bullets = parse_bullets(text)
    bullets = [b.replace("**", "").replace("-", "") for b in bullets]
    return bullets

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

def parse_axes(text: str):
    """
    Parse axes from text.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
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
        return pd.Series({'name': vibe_text, 'high_desc': "", 'low_desc': ""})

    parts = vibe_text.split("High:")
    name = parts[0].strip(": ")
    high_low_parts = parts[1].split("Low:")
    high_desc = high_low_parts[0].strip()
    low_desc = high_low_parts[1].strip()
    return pd.Series({'name': name, 'high_desc': high_desc, 'low_desc': low_desc})

def train_and_evaluate_model(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                           split_train_test:bool=True, solver: str='elasticnet', 
                           n_splits: int=5):
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
    if solver == 'standard':
        model = LogisticRegression(penalty='l2', random_state=42)
    elif solver == 'lasso':
        model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    elif solver == 'elasticnet':
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42)
    else:
        raise ValueError("solver must be one of: 'standard', 'lasso', 'elasticnet'")

    # Initialize arrays to store results across splits
    split_accuracies = []
    split_coefs = []
    
    for split in range(n_splits):
        # Split and train
        if split_train_test:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42+split)
        else:
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
    mse = np.sum((predictions - y_train) ** 2) / (len(y_train) - X_with_intercept.shape[1])
    var_covar_matrix = mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
    standard_errors = np.sqrt(np.diag(var_covar_matrix))[1:]
    z_scores = model.coef_[0] / standard_errors
    p_values = 2 * (1 - stats.norm.cdf(abs(z_scores)))

    # Create dataframe with feature names, coefficients, and p-values
    coef_df = pd.DataFrame({
        "vibe": feature_names,
        "coef": model.coef_[0],
        "p_value": p_values,
    })

    # If using multiple splits, add split-based confidence intervals
    if n_splits > 1:
        coef_df["coef_std"] = np.std(split_coefs, axis=0)
        coef_df["coef_lower_split"] = model.coef_[0] - 1.96 * coef_df["coef_std"]
        coef_df["coef_upper_split"] = model.coef_[0] + 1.96 * coef_df["coef_std"]

    return model, coef_df, accuracy, acc_std

def get_feature_df(vibe_df, split="train"):
    """
    Given a vibe_df with "score" columns pivoted by "vibe", construct X, y
    arrays for preference and identity classification.
    """
    # Pivot to create wide-format scores for each vibe
    feature_df = pd.pivot_table(
        vibe_df,
        values='score',
        index=vibe_df.index,
        columns='vibe',
        fill_value=0
    )
    print(feature_df.columns)

    feature_df_1 = feature_df.copy()
    feature_df_2 = -1 * feature_df.copy()
    
    # Preference data
    X_pref = np.vstack([feature_df_1.to_numpy(), feature_df_2.to_numpy()])
    y_pref = np.concatenate([
        vibe_df["preference_feature"][:len(feature_df)].to_numpy(),
        -1 * vibe_df["preference_feature"][:len(feature_df)].to_numpy()
    ])

    # Model identity data
    y_identity = np.concatenate([
        np.ones(len(feature_df_1)), 
        -1 * np.ones(len(feature_df_2))
    ])

    return feature_df, X_pref, y_pref, y_identity