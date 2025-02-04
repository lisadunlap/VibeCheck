from typing import List
import pandas as pd
import wandb
import re
from omegaconf import OmegaConf

def ranker_postprocess(output: str, models: List[str]) -> str:
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

# class VibeRankerBase:
#     """
#     A class for scoring vibes. 
#     """
#     from omegaconf import OmegaConf
#     def __init__(self, config: OmegaConf):
#         self.config = config

#     def score(self, vibes: List[str], df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
#         """
#         Scores the given vibe.
#         """
#         pass

#     def score_single_vibe(self, vibe: str, df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
#         """
#         Scores the given vibe.
#         """
#         pass

# class VibeRanker(VibeRankerBase):
#     """
#     Uses LLM to rate each response pair as either A, B, or equal.
#     """
#     from components.prompts.ranker_prompts import judge_prompt
#     def __init__(self, config: OmegaConf):
#         super().__init__(config)
#         self.single_position_rank = config["ranker"].get("single_position_rank", False)
        
#     def score(self, vibes: List[str], df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
#         """
#         Scores the given vibe.
#         """
#         pass

def build_vibe_df(vibes: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """Helper: Build vibe_df by concatenating df copies for each vibe."""
    vibe_dfs = []
    for vibe in vibes:
        vibe_df = df.copy()
        vibe_df["vibe"] = vibe
        vibe_dfs.append(vibe_df)

    combined = pd.concat(vibe_dfs).reset_index(drop=True)
    # Drop duplicate columns
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined

def setup_ranker_inputs(
    vibe_df: pd.DataFrame,
    models: List[str],
    single_position_rank: bool
) -> pd.DataFrame:
    """Helper: sets up ranker_inputs, ranker_inputs_reversed, and score_pos_model."""
    import numpy as np

    if single_position_rank:
        # randomly permute models in each row
        rand_models = []
        for i in range(len(vibe_df)):
            np.random.seed(i)
            rand_models.append(list(np.random.permutation(models)))

        vibe_df["score_pos_model"] = rand_models
        ranker_inputs = []
        for i in range(len(vibe_df)):
            row = vibe_df.iloc[i]
            modelA, modelB = rand_models[i]
            ranker_inputs.append(
                f"\nProperty: {row['vibe']}\n\n"
                f"User prompt:\n{row['question']}\n\n"
                f"Response A:\n{row[modelA]}\n\n"
                f"Response B:\n{row[modelB]}\n\n"
                f"Property (restated): {row['vibe']}"
            )
        vibe_df["ranker_inputs"] = ranker_inputs

    else:
        vibe_df["score_pos_model"] = [models for _ in range(len(vibe_df))]
        vibe_df["ranker_inputs"] = vibe_df.apply(
            lambda row: (
                f"Property: {row['vibe']}\n"
                f"User prompt:\n{row['question']}\n\n"
                f"Response A:\n{row[models[0]]}\n\n"
                f"Response B:\n{row[models[1]]}\n\n"
                f"Property (restated): {row['vibe']}"
            ),
            axis=1
        )
        vibe_df["ranker_inputs_reversed"] = vibe_df.apply(
            lambda row: (
                f"Property: {row['vibe']}\n"
                f"User prompt:\n{row['question']}\n\n"
                f"Response A:\n{row[models[1]]}\n\n"
                f"Response B:\n{row[models[0]]}\n\n"
                f"Property (restated): {row['vibe']}"
            ),
            axis=1
        )

    return vibe_df

def merge_ranker_output(
    vibe_df: pd.DataFrame,
    ranker_df: pd.DataFrame,
    models: List[str],
    suffix: str
) -> pd.DataFrame:
    """
    Helper: merges the ranker output (sem_map results) 
    into vibe_df, and applies the ranker_postprocess function.
    """
    # Merge
    vibe_df = vibe_df.merge(
        ranker_df[
            [
                "vibe",
                "question",
                models[0],
                models[1],
                "preference",
                suffix,
                f"raw_output{suffix}",
            ]
        ],
        on=["vibe", "question", models[0], models[1], "preference"],
        how="left",
    )
    vibe_df[suffix] = [
        str(ranker_postprocess(output, model))
        for output, model in zip(vibe_df[suffix], vibe_df["score_pos_model"])
    ]
    return vibe_df

from components.prompts.ranker_prompts import judge_prompt
def rank_vibes(
    vibes: List[str],
    df: pd.DataFrame,
    models: List[str],
    single_position_rank: bool = False,
) -> pd.DataFrame:
    """
    Ranks the two model outputs across the given vibes (axes) using LOTUS ranking prompts.
    """
    ranker_prompt1 = (
        judge_prompt
        + """
Here is the property and the two responses:
{ranker_inputs}
Remember to be as objective as possible and strictly adhere to the response format.
"""
    )
    ranker_prompt2 = (
        judge_prompt
        + """
Here is the property and the two responses:
{ranker_inputs_reversed}
Remember to be as objective as possible and strictly adhere to the response format.
"""
    )

    # 1) Build vibe DataFrame
    vibe_df = build_vibe_df(vibes, df)

    # 2) Create ranker inputs (handles single or multi-position rank creation)
    vibe_df = setup_ranker_inputs(vibe_df, models, single_position_rank)

    # 3) Perform ranker_1
    ranker_1 = vibe_df.sem_map(
        ranker_prompt1, return_raw_outputs=True, suffix="ranker_output_1"
    )
    vibe_df = merge_ranker_output(vibe_df, ranker_1, models, "ranker_output_1")

    # 4) If single_position_rank == False, don't shuffle the models, just run twice
    # TODO: change to not combine scores here, just put all into the regression
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
        vibe_df["ranker_output_2"] = [
            ranker_postprocess(output, model)
            for output, model in zip(
                vibe_df["ranker_output_2"], vibe_df["score_pos_model"]
            )
        ]

        # 5) Compute final score (excluding collisions)
        vibe_df["score"] = vibe_df["ranker_output_1"].apply(
            lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
        )
        vibe_df["score_reversed"] = vibe_df["ranker_output_2"].apply(
            lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
        )
        vibe_df["position_matters"] = (
            (vibe_df["score"] != -1 * vibe_df["score_reversed"])
            | (vibe_df["score"] == 0)
            | (vibe_df["score_reversed"] == 0)
        )
        vibe_df["score"] = vibe_df.apply(
            lambda row: row["score"] if not row["position_matters"] else 0, axis=1
        )
        wandb.summary["prop_position_collisions"] = vibe_df["position_matters"].mean()

    else:
        # single-position rank scoring
        vibe_df["score_label"] = vibe_df["ranker_output_1"]
        vibe_df["score"] = vibe_df["score_label"].apply(
            lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
        )

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
