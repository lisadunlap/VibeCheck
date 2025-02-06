from typing import List
import pandas as pd
import numpy as np
import wandb
import re
from omegaconf import OmegaConf
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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

class VibeRankerBase:
    """
    A class for scoring vibes. 
    """
    from omegaconf import OmegaConf
    def __init__(self, config: OmegaConf):
        self.config = config

    def score(self, vibes: List[str], df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
        """
        Scores the given vibes and returns a DataFrame with the scores per vibe.
        """
        pass

    def score_single_vibe(self, vibe: str, df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
        """
        Scores a given vibe.
        """
        pass

class VibeRanker(VibeRankerBase):
    """
    Uses LLM to rate each response pair as either A, B, or equal.
    """
    from components.prompts.ranker_prompts import judge_prompt
    def __init__(self, config: OmegaConf):
        super().__init__(config)
        self.single_position_rank = config["ranker"].get("single_position_rank", False)
        self.models = list(config["models"])

    def score(
        self, 
        vibes: List[str], 
        df: pd.DataFrame, 
        single_position_rank: bool = None
    ) -> pd.DataFrame:
        """
        Ranks the two model outputs across the given vibes (axes) using LOTUS ranking prompts.
        If single_position_rank is not provided, it will default to the config value.
        """
        single_position_rank = single_position_rank if single_position_rank is not None else self.single_position_rank
        models = self.models
        
        scored_dfs = []
        all_num_collisions = []
        for vibe in vibes:
            df_copy = df.copy()
            df_copy["vibe"] = vibe
            scored_df, num_collisions = self.score_single_vibe(vibe, df_copy, models, single_position_rank)
            scored_dfs.append(scored_df)
            all_num_collisions.append(num_collisions)

        wandb.summary["prop_position_collisions"] = np.mean(all_num_collisions)

        return pd.concat(scored_dfs)

    def score_single_vibe(
        self, 
        vibe: str, 
        df: pd.DataFrame, 
        models: List[str],
        single_position_rank: bool = None
    ) -> pd.DataFrame:
        """
        Scores a single vibe using LOTUS ranking prompts.
        """
        def create_ranker_prompt(inputs_key: str) -> str:
            return (
                self.judge_prompt
                + f"""
Here is the property and the two responses:
{{{inputs_key}}}
Remember to be as objective as possible and strictly adhere to the response format.
"""
            )

        ranker_prompt1 = create_ranker_prompt("ranker_inputs")
        ranker_prompt2 = create_ranker_prompt("ranker_inputs_reversed")

        # Create ranker inputs (handles single or multi-position rank creation)
        vibe_df = setup_ranker_inputs(df, models, single_position_rank)

        # Perform ranker_1
        ranker_1 = vibe_df.sem_map(
            ranker_prompt1, return_raw_outputs=True, suffix="ranker_output_1"
        )
        vibe_df = merge_ranker_output(vibe_df, ranker_1, models, "ranker_output_1")

        # If single_position_rank == False, don't shuffle the models, just run twice
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

            # Compute final score (excluding collisions)
            vibe_df["score"] = vibe_df["ranker_output_1"].apply(
                lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
            )
            vibe_df["score_reversed"] = vibe_df["ranker_output_2"].apply(
                lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
            )
            vibe_df["position_matters"] = (
                (vibe_df["score"] != -1 * vibe_df["score_reversed"])
                & (vibe_df["score"] != 0)
                & (vibe_df["score_reversed"] != 0)
            )
            vibe_df["score"] = vibe_df.apply(
                lambda row: row["score"] if not row["position_matters"] else 0, axis=1
            )
            wandb.summary[f"prop_position_collisions_{vibe}"] = vibe_df["position_matters"].mean()
            num_collisions = vibe_df["position_matters"].mean()
        else:
            # single-position rank scoring
            vibe_df["score_label"] = vibe_df["ranker_output_1"]
            vibe_df["score"] = vibe_df["score_label"].apply(
                lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
            )
            num_collisions = 0

        return vibe_df, num_collisions

# def build_vibe_df(vibes: List[str], df: pd.DataFrame) -> pd.DataFrame:
#     """Helper: Build vibe_df by concatenating df copies for each vibe."""
#     vibe_dfs = []
#     for vibe in vibes:
#         vibe_df = df.copy()
#         vibe_df["vibe"] = vibe
#         vibe_dfs.append(vibe_df)

#     combined = pd.concat(vibe_dfs).reset_index(drop=True)
#     # Drop duplicate columns
#     combined = combined.loc[:, ~combined.columns.duplicated()]
#     return combined

def setup_ranker_inputs(
    vibe_df: pd.DataFrame,
    models: List[str],
    single_position_rank: bool
) -> pd.DataFrame:
    """Helper: sets up ranker_inputs, ranker_inputs_reversed, and score_pos_model."""

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

# from components.prompts.ranker_prompts import judge_prompt
# def rank_vibes(
#     vibes: List[str],
#     df: pd.DataFrame,
#     models: List[str],
#     single_position_rank: bool = False,
# ) -> pd.DataFrame:
#     """
#     Ranks the two model outputs across the given vibes (axes) using LOTUS ranking prompts.
#     """
#     ranker_prompt1 = (
#         judge_prompt
#         + """
# Here is the property and the two responses:
# {ranker_inputs}
# Remember to be as objective as possible and strictly adhere to the response format.
# """
#     )
#     ranker_prompt2 = (
#         judge_prompt
#         + """
# Here is the property and the two responses:
# {ranker_inputs_reversed}
# Remember to be as objective as possible and strictly adhere to the response format.
# """
#     )

#     # 1) Build vibe DataFrame
#     vibe_df = build_vibe_df(vibes, df)

#     # 2) Create ranker inputs (handles single or multi-position rank creation)
#     vibe_df = setup_ranker_inputs(vibe_df, models, single_position_rank)

#     # 3) Perform ranker_1
#     ranker_1 = vibe_df.sem_map(
#         ranker_prompt1, return_raw_outputs=True, suffix="ranker_output_1"
#     )
#     vibe_df = merge_ranker_output(vibe_df, ranker_1, models, "ranker_output_1")

#     # 4) If single_position_rank == False, don't shuffle the models, just run twice
#     # TODO: change to not combine scores here, just put all into the regression
#     if not single_position_rank:
#         ranker_2 = vibe_df.sem_map(
#             ranker_prompt2, return_raw_outputs=True, suffix="ranker_output_2"
#         )
#         vibe_df = vibe_df.merge(
#             ranker_2[
#                 ["question", models[0], models[1], "preference", "ranker_output_2"]
#             ],
#             on=["question", models[0], models[1], "preference"],
#             how="left",
#         )
#         vibe_df["ranker_output_2"] = [
#             ranker_postprocess(output, model)
#             for output, model in zip(
#                 vibe_df["ranker_output_2"], vibe_df["score_pos_model"]
#             )
#         ]

#         # 5) Compute final score (excluding collisions)
#         vibe_df["score"] = vibe_df["ranker_output_1"].apply(
#             lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
#         )
#         vibe_df["score_reversed"] = vibe_df["ranker_output_2"].apply(
#             lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
#         )
#         vibe_df["position_matters"] = (
#             (vibe_df["score"] != -1 * vibe_df["score_reversed"])
#             | (vibe_df["score"] == 0)
#             | (vibe_df["score_reversed"] == 0)
#         )
#         vibe_df["score"] = vibe_df.apply(
#             lambda row: row["score"] if not row["position_matters"] else 0, axis=1
#         )
#         wandb.summary["prop_position_collisions"] = vibe_df["position_matters"].mean()

#     else:
#         # single-position rank scoring
#         vibe_df["score_label"] = vibe_df["ranker_output_1"]
#         vibe_df["score"] = vibe_df["score_label"].apply(
#             lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
#         )

#     return vibe_df


from components.utils_llm import get_llm_embedding
from sklearn.metrics.pairwise import cosine_similarity
class VibeRankerEmbedding(VibeRanker):
    """
    Uses embedding similarity to score each response pair.
    """
    def __init__(self, config: OmegaConf):
        super().__init__(config)
        self.embedding_model = config["ranker"]["embedding_model"]

    def score_single_vibe(self, vibe: str, df: pd.DataFrame, models: List[str], single_position_rank: bool = None) -> tuple[pd.DataFrame, float]:
        """
        Scores a single vibe using embedding similarity.
        Returns tuple of (scored_dataframe, num_collisions=0) to match parent class interface.
        """
        
        df = df.copy()
        df["model_a_embedding"] = df[models[0]].apply(lambda x: np.array(get_llm_embedding(x, self.embedding_model)))
        df["model_b_embedding"] = df[models[1]].apply(lambda x: np.array(get_llm_embedding(x, self.embedding_model)))
        df["vibe_embedding"] = df["vibe"].apply(lambda x: np.array(get_llm_embedding(x, self.embedding_model)))
        df["model_a_embedding"] = df["model_a_embedding"].apply(lambda x: x / np.linalg.norm(x))
        df["model_b_embedding"] = df["model_b_embedding"].apply(lambda x: x / np.linalg.norm(x))
        df["vibe_embedding"] = df["vibe_embedding"].apply(lambda x: x / np.linalg.norm(x))

        # Calculate similarities
        df["model_a_vibe_sim"] = df.apply(
            lambda row: cosine_similarity(
                row["model_a_embedding"].reshape(1, -1), 
                row["vibe_embedding"].reshape(1, -1)
            )[0][0],
            axis=1,
        )
        df["model_b_vibe_sim"] = df.apply(
            lambda row: cosine_similarity(
                row["model_b_embedding"].reshape(1, -1), 
                row["vibe_embedding"].reshape(1, -1)
            )[0][0],
            axis=1,
        )

        self.plot_embedding_similarity(vibe, df, models)

        # Calculate scores
        df["score"] = df.apply(
            lambda row: 1 if row["model_a_vibe_sim"] > row["model_b_vibe_sim"] else -1,
            axis=1,
        )

        # Drop embedding columns
        df = df.drop(columns=["model_a_embedding", "model_b_embedding", "vibe_embedding"])

        df["score_pos_model"] = [models for _ in range(len(df))] # hack: need this for training prediction models
        train_embedding_model(df, models, vibe, self.embedding_model, self.config)
        return df, 0  
    
    @staticmethod
    def plot_embedding_similarity(vibe: str, df: pd.DataFrame, models: List[str]) -> None:
        """
        Plots the embedding similarity distribution for a given vibe.
        """
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["model_a_vibe_sim"],
            name=models[0],
            opacity=0.75,
            nbinsx=30
        ))
        fig.add_trace(go.Histogram(
            x=df["model_b_vibe_sim"], 
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


class VibeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(np.stack(embeddings))
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class VibePredictor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)  # 3 classes: -1, 0, 1
        )
    
    def forward(self, x):
        return self.model(x)

def train_embedding_model(df: pd.DataFrame, models: List[str], vibe: str, embedding_model: str, config: OmegaConf) -> None:
    """
    Trains an embedding model on the given data using PyTorch. Ranks half the data using the LLM ranker,
    trains a neural network to predict the ranker score, then uses it to predict the other half.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    df = df.copy()
    ranker = VibeRanker(config)
    df, _ = ranker.score_single_vibe(vibe, df, models, single_position_rank=True)
    
    # Get embeddings
    df["model_a_embedding"] = df[models[0]].apply(lambda x: np.array(get_llm_embedding(x, embedding_model)))
    df["model_b_embedding"] = df[models[1]].apply(lambda x: np.array(get_llm_embedding(x, embedding_model)))
    df["vibe_embedding"] = df["vibe"].apply(lambda x: np.array(get_llm_embedding(x, embedding_model)))
    
    # Normalize embeddings
    df["model_a_embedding"] = df["model_a_embedding"].apply(lambda x: x / np.linalg.norm(x))
    df["model_b_embedding"] = df["model_b_embedding"].apply(lambda x: x / np.linalg.norm(x))
    df["vibe_embedding"] = df["vibe_embedding"].apply(lambda x: x / np.linalg.norm(x))
    
    # Calculate response difference embedding
    df["response_diff_embedding"] = df["model_a_embedding"] - df["model_b_embedding"]
    df["response_diff_embedding"] = df["response_diff_embedding"].apply(lambda x: x / np.linalg.norm(x))
    
    # Split data
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)
    
    # Create datasets
    train_dataset = VibeDataset(df_train["response_diff_embedding"].values, df_train["score"].values + 1)  # Shift scores to 0,1,2
    test_dataset = VibeDataset(df_test["response_diff_embedding"].values, df_test["score"].values + 1)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    embedding_dim = len(df_train["response_diff_embedding"].iloc[0])
    model = VibePredictor(embedding_dim).to(device)
    
    # Training setup
    # Calculate class weights
    labels = df_train["score"].values + 1  # Shift scores to 0,1,2
    class_counts = np.bincount(labels)
    class_weights = torch.FloatTensor(len(class_counts) / class_counts).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_embeddings, _ in test_loader:
            batch_embeddings = batch_embeddings.to(device)
            outputs = model(batch_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # Convert predictions back to original scale (-1, 0, 1)
    df_test["score_pred"] = [pred - 1 for pred in predictions]
    # log distribution of predictions
    wandb.log({f"vibe_predictor_pred_dist_{vibe}": wandb.Histogram(df_test["score_pred"])})
    
    # Calculate and log accuracy
    accuracy = (df_test["score"] == df_test["score_pred"]).mean()
    wandb.log({f"vibe_predictor_accuracy_{vibe}": accuracy})
    
    return df_test


# from components.utils_llm import get_llm_embedding
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# def rank_vibes_embedding(vibes: List[str], 
#                          df: pd.DataFrame, 
#                          models: List[str],
#                          embedding_model: str = "text-embedding-3-small") -> pd.DataFrame:
#     """
#     Ranks the two model outputs across the given vibes (axes) using embedding similarity.
#     This is much cheaper than the LLM ranker, but way less accurate. Maybe we can train an MLP with some LLM labels.
#     Also maybe im not normalizing the embeddings correctly.
#     """
#     import plotly.graph_objects as go

#     vibe_dfs = []
#     for vibe in vibes:
#         vibe_df = df.copy()
#         vibe_df["vibe"] = vibe
#         vibe_dfs.append(vibe_df)

#     # drop any duplicate columns
#     vibe_df = vibe_df.loc[:, ~vibe_df.columns.duplicated()]
#     vibe_df["model_a_embedding"] = vibe_df[models[0]].apply(lambda x: np.array(get_llm_embedding(x, embedding_model)))
#     vibe_df["model_b_embedding"] = vibe_df[models[1]].apply(lambda x: np.array(get_llm_embedding(x, embedding_model)))
#     vibe_df["vibe_embedding"] = vibe_df["vibe"].apply(lambda x: np.array(get_llm_embedding(x, embedding_model)))

#     vibe_df["model_a_embedding"] = vibe_df["model_a_embedding"].apply(lambda x: x / np.linalg.norm(x))
#     vibe_df["model_b_embedding"] = vibe_df["model_b_embedding"].apply(lambda x: x / np.linalg.norm(x))
#     vibe_df["vibe_embedding"] = vibe_df["vibe_embedding"].apply(lambda x: x / np.linalg.norm(x))

#     # vibe_embeddings = vibe_df.drop_duplicates(subset=["vibe"])["vibe_embedding"].values
#     # model_a_embeddings = vibe_df.drop_duplicates(subset=["question"])["model_a_embedding"].values
#     # model_b_embeddings = vibe_df.drop_duplicates(subset=["question"])["model_b_embedding"].values

#     # # normalize across all embeddings
#     # vibe_df["model_a_embedding"] = vibe_df["model_a_embedding"].apply(lambda x: x - np.mean(all_embeddings, axis=0))
#     # vibe_df["model_b_embedding"] = vibe_df["model_b_embedding"].apply(lambda x: x - np.mean(all_embeddings, axis=0))
#     # vibe_df["vibe_embedding"] = vibe_df["vibe_embedding"].apply(lambda x: x - np.mean(vibe_embeddings, axis=0))
  
#     vibe_df["model_a_vibe_sim"] = vibe_df.apply(
#         lambda row: cosine_similarity(
#             row["model_a_embedding"].reshape(1, -1), 
#             row["vibe_embedding"].reshape(1, -1)
#         )[0][0],  # Extract scalar value from 2D result
#         axis=1,
#     )
#     vibe_df["model_b_vibe_sim"] = vibe_df.apply(
#         lambda row: cosine_similarity(
#             row["model_b_embedding"].reshape(1, -1), 
#             row["vibe_embedding"].reshape(1, -1)
#         )[0][0],  # Extract scalar value from 2D result
#         axis=1,
#     )

#     # Create and log histogram plots for each vibe
#     for vibe in vibes:
#         vibe_subset = vibe_df[vibe_df["vibe"] == vibe]
        
#         fig = go.Figure()
#         fig.add_trace(go.Histogram(
#             x=vibe_subset["model_a_vibe_sim"],
#             name=models[0],
#             opacity=0.75,
#             nbinsx=30
#         ))
#         fig.add_trace(go.Histogram(
#             x=vibe_subset["model_b_vibe_sim"], 
#             name=models[1],
#             opacity=0.75,
#             nbinsx=30
#         ))
        
#         fig.update_layout(
#             title=f"Embedding Similarity Distribution for<br><span style='font-size:80%'>'{vibe}'</span>",
#             xaxis_title="Cosine Similarity",
#             yaxis_title="Count",
#             barmode='overlay',
#             template="plotly_white"
#         )
        
#         truncated_vibe = ' '.join(vibe.split()[:3])
#         wandb.log({f"Vibe Scoring/embedding_sim_dist_{truncated_vibe}": wandb.Html(fig.to_html())})

#     # if cos sim between model_a_embeddings and vibe_embeddings is greater than model_b_embeddings and vibe_embeddings, then 1, else -1
#     vibe_df["score"] = vibe_df.apply(
#         lambda row: 1 if row["model_a_vibe_sim"] > row["model_b_vibe_sim"] else -1,
#         axis=1,
#     )
    
#     # drop embeddings
#     vibe_df = vibe_df.drop(columns=["model_a_embedding", "model_b_embedding", "vibe_embedding"])
#     return vibe_df
