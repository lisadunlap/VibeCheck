#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    """
    Create the prompt for reducing axes.
    """
    return f"""Below is a list of axes with a description of what makes a piece of text low or high on this axis. I would like to summarize this list to at most {num_reduced_axes} representative axes.

Here is the list of axes:
{{differences}}

These axes should contain only one concept and should be human interpretable. Some examples of bad axes include:
- "Configuration Clarity: High: Clearly defined structure and purpose. Low: Vaguely defined, minimal purpose."
- "Language and Communication: High: Varied/precise, complex structure. Low: Straightforward, simple or general language."
- "Content Quality: High: High quality, engaging, informative. Low: Low quality, unengaging, uninformative."

Some examples of good axes include:
- "Complexity: High: Complex, multi-layered, intricate. Low: Simple, straightforward, easy to understand."
- "Efficiency (coding): High: Code optimized for runtime, minimal memory usage. Low: Code inefficient, high memory usage."

Please return the simplified list of <={num_reduced_axes} axes with any redundant axes removed and the descriptions of what makes a piece of text low or high on this axis simplified.

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

def rank_axes(vibes, df, models, lm, rm):
    """
    Ranks the two model outputs across the given vibes (axes) using LOTUS ranking prompts.
    """
    import re
    import lotus
    from lotus.cache import CacheConfig, CacheType, CacheFactory

    def ranker_postprocess(output: str) -> int:
        """
        Postprocess the ranker's output to extract whether model A is favored (1), B is favored (-1), or tie/NA (0).
        """
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
{ranker_inputs_1}

Remember to be as objective as possible and strictly adhere to the response format.
"""

    ranker_prompt2 = judge_systems_prompt + """
Here is the property and the two responses:
{ranker_inputs_2}

Remember to be as objective as possible and strictly adhere to the response format.
"""

    vibe_dfs = []
    for vibe in vibes:
        vibe_df = df.copy()
        vibe_df["vibe"] = vibe
        vibe_dfs.append(vibe_df)

    vibe_df = pd.concat(vibe_dfs)

    # drop any duplicate columns
    vibe_df = vibe_df.loc[:, ~vibe_df.columns.duplicated()]
    vibe_df["ranker_inputs_1"] = vibe_df.apply(
        lambda row: f"Property: {row['vibe']}\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[models[0]]}\n\nResponse B:\n{row[models[1]]}",
        axis=1
    )
    vibe_df["ranker_inputs_2"] = vibe_df.apply(
        lambda row: f"Property: {row['vibe']}\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[models[1]]}\n\nResponse B:\n{row[models[0]]}",
        axis=1
    )

    # Use lotus sem_map
    ranker_1 = vibe_df.sem_map(ranker_prompt1, return_raw_outputs=True, suffix="ranker_output_1")
    ranker_2 = vibe_df.sem_map(ranker_prompt2, return_raw_outputs=True, suffix="ranker_output_2")

    vibe_df = pd.concat([vibe_df, ranker_1, ranker_2], axis=1)
    vibe_df = vibe_df.loc[:, ~vibe_df.columns.duplicated()]

    # Postprocess
    vibe_df["ranker_output_1"] = vibe_df["ranker_output_1"].apply(ranker_postprocess)
    vibe_df["ranker_output_2"] = vibe_df["ranker_output_2"].apply(ranker_postprocess)

    # Detect if position matters
    vibe_df["position_matters"] = vibe_df["ranker_output_1"] != -1 * vibe_df["ranker_output_2"]
    vibe_df["score"] = vibe_df.apply(
        lambda row: row["ranker_output_1"] if not row["position_matters"] else 0, 
        axis=1
    )

    return vibe_df

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

def train_and_evaluate_model(X, y, feature_names, split_train_test=True):
    """
    Train a logistic regression model on (X, y) and compute accuracy and p-values for each feature.
    """
    # Split and train
    if split_train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    else:
        X_train = X
        X_test = X
        y_train = y
        y_test = y
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {accuracy}")

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
        "p_value": p_values
    })

    return model, coef_df, accuracy

# def train_and_evaluate_model(X, y, feature_names, model_name=""):
#     """
#     Train a ridge regression model on (X, y) and compute R² and p-values for each feature.
#     """
#     model = Ridge(alpha=1.0)  # alpha is the regularization strength
#     model.fit(X, y)

#     # Calculate R² score
#     r2_score = model.score(X, y)
#     print(f"{model_name} R² Score: {r2_score}")

#     # Calculate p-values
#     n = X.shape[0]
#     p = X.shape[1]
    
#     # Calculate MSE and dof
#     y_pred = model.predict(X)
#     mse = np.sum((y - y_pred) ** 2) / (n - p - 1)
    
#     # Calculate variance-covariance matrix
#     X_normalized = X - X.mean(axis=0)
#     var_covar_matrix = mse * np.linalg.inv(X_normalized.T @ X_normalized + model.alpha * np.eye(p))
    
#     # Calculate standard errors and t-statistics
#     standard_errors = np.sqrt(np.diag(var_covar_matrix))
#     t_stats = model.coef_ / standard_errors
    
#     # Calculate p-values using t-distribution
#     p_values = 2 * (1 - stats.t.cdf(abs(t_stats), df=n-p-1))

#     # Create dataframe with feature names, coefficients, and p-values
#     coef_df = pd.DataFrame({
#         "vibe": feature_names,
#         "coef": model.coef_,
#         "p_value": p_values
#     })

#     return model, coef_df, r2_score

def get_feature_df(vibe_df, split="train"):
    """
    Given a vibe_df with "score" columns pivoted by "vibe", construct X, y
    arrays for preference and identity classification.
    """
    # Pivot to create wide-format scores for each vibe
    feature_df = pd.pivot_table(
        vibe_df[vibe_df["split"] == split],
        values='score',
        index=vibe_df[vibe_df["split"] == split].index,
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