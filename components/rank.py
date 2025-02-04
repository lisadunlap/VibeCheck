from typing import List
import pandas as pd
import wandb
import re


def ranker_postprocess(output: str, models: List[str]) -> str:
    """
    Postprocess the ranker's output to extract whether model A is favored (1), B is favored (-1), or tie/NA (0).
    """
    try:
        output = output.replace("Output ", "").replace("output ", "")
        output = re.sub(r"[#*]", "", output)
        score_pattern = re.compile(r"Model: (A|B|N/A|unsure|equal)", re.I | re.M)
        score = score_pattern.findall(output)
        if not score:
            return "tie"
        if score[0].lower() == "a":
            return models[0]
        elif score[0].lower() == "b":
            return models[1]
        else:
            return "tie"
    except Exception as e:
        print(f"Error in ranker_postprocess: {output}\n\n{e}")
        return "tie"


def convert_scores(scores: List[str], original_models: List[str]) -> List[int]:
    return [1 if score == original_models[0] else -1 if score == original_models[1] else 0 for score in scores]

def rank_vibes(
    vibes: List[str],
    df: pd.DataFrame,
    models: List[str],
    single_position_rank: bool = False,
) -> pd.DataFrame:
    """
    Ranks the two model outputs across the given vibes (axes) using LOTUS ranking prompts.
    """
    judge_systems_prompt = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a given propoery. Which repose better aligns more with the given property, A, B, or equal?

Your sole focus is to determine which response better aligns with the given property, NOT how good or bad the response is. Do NOT let the position of the model outputs influence your decision and remain as objective as possible. Consider what the property means and how it applies to the outputs. Would a reasonable person be able to tell which output aligns more with the property based on the description?

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
    if single_position_rank:
        # randomly shuffle the models for each row 
        rand_models = []
        for i in range(len(vibe_df)):
            np.random.seed(i)
            rand_models.append(list(np.random.permutation(models)))
        np.random.seed(42)
        print(rand_models[:5])
        vibe_df["score_pos_model"] = rand_models
        ranker_inputs = []
        for i in range(len(vibe_df)):
            ranker_inputs.append(f"\nProperty: {vibe_df['vibe'][i]}\n\nUser prompt:\n{vibe_df['question'][i]}\n\nResponse A:\n{vibe_df[rand_models[i][0]][i]}\n\nResponse B:\n{vibe_df[rand_models[i][1]][i]}\n\nProperty (restated): {vibe_df['vibe'][i]}")
        vibe_df["ranker_inputs"] = ranker_inputs
    # vibe_df["ranker_inputs"] = vibe_df.apply(
    #     lambda row: f"\nProperty: {row['vibe']}\n\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[models[0]]}\n\nResponse B:\n{row[models[1]]}\n\nProperty (restated): {row['vibe']}",
    #     axis=1,
    # )

    if not single_position_rank:
        vibe_df["score_pos_model"] = [models for _ in range(len(vibe_df))]
        vibe_df["ranker_inputs"] = vibe_df.apply(
            lambda row: f"Property: {row['vibe']}\nUser prompt:\n{row['question']}\n\nResponse A:\n{row[models[0]]}\n\nResponse B:\n{row[models[1]]}\n\nProperty (restated): {row['vibe']}",
            axis=1,
        )
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
    vibe_df["ranker_output_1"] = [str(ranker_postprocess(output, model)) for output, model in zip(vibe_df["ranker_output_1"], vibe_df["score_pos_model"])]

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
        vibe_df["ranker_output_2"] = [ranker_postprocess(output, model) for output, model in zip(vibe_df["ranker_output_2"], vibe_df["score_pos_model"])]
        vibe_df["score"] = vibe_df["ranker_output_1"].apply(lambda x: 1 if x == models[0] else -1 if x == models[1] else 0)
        vibe_df["score_reversed"] = vibe_df["ranker_output_2"].apply(lambda x: 1 if x == models[0] else -1 if x == models[1] else 0)
        vibe_df["position_matters"] = (
            (vibe_df["score"] != -1 * vibe_df["score_reversed"]) | 
            (vibe_df["score"] == 0) | 
            (vibe_df["score_reversed"] == 0)
        )
        vibe_df["score"] = vibe_df.apply(
            lambda row: row["score"] if not row["position_matters"] else 0,
            axis=1,
        )
        wandb.summary["prop_position_collisions"] = vibe_df["position_matters"].mean()
    else:
        vibe_df["score_label"] = vibe_df["ranker_output_1"]
        vibe_df["score"] = vibe_df["score_label"].apply(lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0))

    return vibe_df


from components.utils_llm import get_llm_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def rank_vibes_embedding(vibes: List[str], 
                         df: pd.DataFrame, 
                         models: List[str],
                         embedding_model: str = "text-embedding-3-small") -> pd.DataFrame:
    """
    Ranks the two model outputs across the given vibes (axes) using embedding similarity.
    This is much cheaper than the LLM ranker, but way less accurate. Maybe we can train an MLP with some LLM labels.
    Also maybe im not normalizing the embeddings correctly.
    """
    import plotly.graph_objects as go

    vibe_dfs = []
    for vibe in vibes:
        vibe_df = df.copy()
        vibe_df["vibe"] = vibe
        vibe_dfs.append(vibe_df)

    # drop any duplicate columns
    vibe_df = vibe_df.loc[:, ~vibe_df.columns.duplicated()]
    vibe_df["model_a_embedding"] = vibe_df[models[0]].apply(lambda x: np.array(get_llm_embedding(x, embedding_model)))
    vibe_df["model_b_embedding"] = vibe_df[models[1]].apply(lambda x: np.array(get_llm_embedding(x, embedding_model)))
    vibe_df["vibe_embedding"] = vibe_df["vibe"].apply(lambda x: np.array(get_llm_embedding(x, embedding_model)))

    vibe_df["model_a_embedding"] = vibe_df["model_a_embedding"].apply(lambda x: x / np.linalg.norm(x))
    vibe_df["model_b_embedding"] = vibe_df["model_b_embedding"].apply(lambda x: x / np.linalg.norm(x))
    vibe_df["vibe_embedding"] = vibe_df["vibe_embedding"].apply(lambda x: x / np.linalg.norm(x))

    vibe_embeddings = vibe_df.drop_duplicates(subset=["vibe"])["vibe_embedding"].values
    model_a_embeddings = vibe_df.drop_duplicates(subset=["question"])["model_a_embedding"].values
    model_b_embeddings = vibe_df.drop_duplicates(subset=["question"])["model_b_embedding"].values

    all_embeddings = np.concatenate([model_a_embeddings, model_b_embeddings])

    # normalize across all embeddings
    vibe_df["model_a_embedding"] = vibe_df["model_a_embedding"].apply(lambda x: x - np.mean(all_embeddings, axis=0))
    vibe_df["model_b_embedding"] = vibe_df["model_b_embedding"].apply(lambda x: x - np.mean(all_embeddings, axis=0))
    vibe_df["vibe_embedding"] = vibe_df["vibe_embedding"].apply(lambda x: x - np.mean(vibe_embeddings, axis=0))
  
    vibe_df["model_a_vibe_sim"] = vibe_df.apply(
        lambda row: cosine_similarity(
            row["model_a_embedding"].reshape(1, -1), 
            row["vibe_embedding"].reshape(1, -1)
        )[0][0],  # Extract scalar value from 2D result
        axis=1,
    )
    vibe_df["model_b_vibe_sim"] = vibe_df.apply(
        lambda row: cosine_similarity(
            row["model_b_embedding"].reshape(1, -1), 
            row["vibe_embedding"].reshape(1, -1)
        )[0][0],  # Extract scalar value from 2D result
        axis=1,
    )

    # Create and log histogram plots for each vibe
    for vibe in vibes:
        vibe_subset = vibe_df[vibe_df["vibe"] == vibe]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vibe_subset["model_a_vibe_sim"],
            name=models[0],
            opacity=0.75,
            nbinsx=30
        ))
        fig.add_trace(go.Histogram(
            x=vibe_subset["model_b_vibe_sim"], 
            name=models[1],
            opacity=0.75,
            nbinsx=30
        ))
        
        fig.update_layout(
            title=f"Embedding Similarity Distribution for<br><span style='font-size:80%'>'{vibe}'</span>",
            xaxis_title="Cosine Similarity",
            yaxis_title="Count",
            barmode='overlay',
            template="plotly_white"
        )
        
        truncated_vibe = ' '.join(vibe.split()[:3])
        wandb.log({f"Vibe Scoring/embedding_sim_dist_{truncated_vibe}": wandb.Html(fig.to_html())})

    # if cos sim between model_a_embeddings and vibe_embeddings is greater than model_b_embeddings and vibe_embeddings, then 1, else -1
    vibe_df["score"] = vibe_df.apply(
        lambda row: 1 if row["model_a_vibe_sim"] > row["model_b_vibe_sim"] else -1,
        axis=1,
    )
    
    # drop embeddings
    vibe_df = vibe_df.drop(columns=["model_a_embedding", "model_b_embedding", "vibe_embedding"])
    return vibe_df
