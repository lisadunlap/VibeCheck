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
from tqdm import tqdm

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.cache import CacheConfig, CacheType, CacheFactory
from components.utils_llm import get_llm_embedding, get_llm_output
import components.prompts.ranker_prompts as ranker_prompts

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
    
def ranker_postprocess_multi(output: str, models: List[str]) -> List[str]:
    """
    Process multi-vibe ranking output into a list of model preferences.
    """
    try:
        output = output.replace("Output ", "").replace("output ", "")
        output = re.sub(r"[#*]", "", output)
        
        # Find the Ranking: section and extract property-based responses
        ranking_pattern = re.compile(r"(?:Ranking:|^)\s*(?:Property\s+\d+:\s*(A|B|N/A|unsure|equal)\s*Analysis:.*(?:\s*\n|$))+", re.I | re.M)
        ranking_section = ranking_pattern.search(output)
        
        if not ranking_section:
            return []
            
        # Extract individual rankings
        score_pattern = re.compile(r"Property\s+\d+:\s*(A|B|N/A|unsure|equal)", re.I)
        scores = score_pattern.findall(ranking_section.group())
        
        # Convert each score to the appropriate model name or "tie"
        results = []
        for score in scores:
            score = score.lower()
            if score == "a":
                results.append(models[0])
            elif score == "b":
                results.append(models[1])
            else:
                results.append("tie")
                
        return results
        
    except Exception as e:
        print(f"Error in ranker_postprocess_multi: {output}\n\n{e}")
        return []

def convert_scores(scores: List[str], original_models: List[str]) -> List[int]:
    return [1 if score == original_models[0] else -1 if score == original_models[1] else 0 for score in scores]

class VibeRankerBase:
    """
    A class for scoring vibes. 
    """
    from omegaconf import OmegaConf
    def __init__(self, config: OmegaConf):
        self.config = config
        # Initialize LOTUS
        cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
        cache = CacheFactory.create_cache(cache_config)
        lm = LM(model=config["ranker"].model, cache=cache)
        lotus.settings.configure(lm=lm, enable_cache=True)

    def score(self, vibes: List[str], df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
        """
        Scores the given vibes and returns a DataFrame with the scores per vibe.
        """
        pass

    def score_vibe(self, vibe: str, df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
        """
        Scores a given vibe/vibes.
        """
        pass

class VibeRanker(VibeRankerBase):
    """
    Uses a batch of vibes to score the models. TOOD: merge with VibeRanker
    """
    def __init__(self, config: OmegaConf):
        super().__init__(config)
        self.batch_size = config["ranker"].get("batch_size", 10)
        self.models = list(config["models"])
        self.vibe_batch_size = config["ranker"].get("vibe_batch_size", 5)  # New parameter

    def score(self, vibes: List[str], df: pd.DataFrame, single_position_rank: bool = None) -> pd.DataFrame:
        """
        Scores the given vibes and returns a DataFrame with the scores per vibe.
        """
        all_scored_dfs = []
        for i in range(0, len(vibes), self.vibe_batch_size):
            vibe_batch = vibes[i:i + self.vibe_batch_size]

            scored_df = self.score_vibe(vibe_batch, df)
            scored_df_reversed = self.score_vibe(vibe_batch, df, reverse_position=True)

            # rename score in scored_to to score_forward and score_reversed to score_backward
            scored_df.rename(columns={"score": "score_forward", "score_reversed": "score_backward"}, inplace=True)
            scored_df_reversed.rename(columns={"score": "score_backward", "score_reversed": "score_forward"}, inplace=True)
            
            # merge the two dataframes on conversation_id, preference, and vibe
            scored_df = scored_df.merge(scored_df_reversed[["conversation_id","vibe","score_backward"]], on=["conversation_id","vibe"], how="inner")
            
            def is_position_bias(item1, item2):
                return item1 == item2 and item1 != 0 and item2 != 0
            scored_df["position_matters"] = scored_df.apply(lambda row: is_position_bias(row["score_forward"], row["score_backward"]), axis=1)
            scored_df["score"] = scored_df.apply(lambda row: row["score_forward"] if not row["position_matters"] else 0, axis=1)
            all_scored_dfs.append(scored_df)

        return pd.concat(all_scored_dfs)

    def score_vibe(self, vibe_batch: List[str], df: pd.DataFrame, reverse_position: bool = False) -> pd.DataFrame:
        """
        Scores a batch of vibes and returns a DataFrame with the scores.
        """
        vibe_df = df.copy()
        models = self.models
        vibe_df["score_pos_model"] = [models for _ in range(len(vibe_df))]

        if reverse_position:
            vibe_df["ranker_inputs"] = vibe_df.apply(
                lambda row: (
                    f"Properties:\n" + "\n".join(f"Property {i+1}: {vibe}" for i, vibe in enumerate(vibe_batch)) + "\n"
                    f"--------------------------------\n\nUser prompt:\n{row['question']}\n\n"
                    f"\n\nResponse A:\n{row[models[1]]}\n\n"
                    f"\n\nResponse B:\n{row[models[0]]}\n\n"
                    f"--------------------------------\n\nProperties (restated):\n" + "\n".join(f"Property {i+1}: {vibe}" for i, vibe in enumerate(vibe_batch)) + "\n"
                ),
                axis=1
            )
        else:
            vibe_df["ranker_inputs"] = vibe_df.apply(
                lambda row: (
                    f"Properties:\n" + "\n".join(f"Property {i+1}: {vibe}" for i, vibe in enumerate(vibe_batch)) + "\n"
                    f"--------------------------------\n\nUser prompt:\n{row['question']}\n\n"
                    f"\n\nResponse A:\n{row[models[0]]}\n\n"
                    f"n\nResponse B:\n{row[models[1]]}\n\n"
                    f"--------------------------------\n\nProperties (restated):\n" + "\n".join(f"Property {i+1}: {vibe}" for i, vibe in enumerate(vibe_batch)) + "\n"
                ),
                axis=1
            )

        ranker_prompt = getattr(ranker_prompts, self.config["ranker"].prompt)
        vibe_df["ranker_output"] = get_llm_output([
                ranker_prompt.format(inputs=vibe_df.iloc[idx]["ranker_inputs"])
                for idx in range(len(vibe_df))
            ], self.config["ranker"].model, cache=True)
        vibe_df["ranker_output"] = [
            ranker_postprocess_multi(output, models)
            for output in vibe_df["ranker_output"]
        ]
        # Check for incorrect outputs and retry
        max_retries = 3
        for retry in range(max_retries):
            retry_indices = []
            for i in range(len(vibe_df)):
                if len(vibe_df["ranker_output"][i]) != len(vibe_batch):
                    retry_indices.append(i)

            if not retry_indices:
                break

            print(f"Retry {retry + 1}: Attempting to fix {len(retry_indices)} incorrect outputs")
            # Prepare prompts for retry
            retry_prompts = [
                getattr(ranker_prompts, self.config["ranker"].prompt).format(inputs=vibe_df.iloc[idx]["ranker_inputs"])
                for idx in retry_indices
            ]

            try:
                # Use get_llm_output with threading for batch processing
                new_outputs = get_llm_output(retry_prompts, "gpt-4o", cache=retry == 0)
                for idx, new_output in zip(retry_indices, new_outputs):
                    vibe_df.at[idx, "ranker_output"] = ranker_postprocess_multi(new_output, models)
            except Exception as e:
                print(f"Error during retry: {e}")

        # Count remaining errors after retries
        final_errors = sum(len(output) != len(vibe_batch) for output in vibe_df["ranker_output"])
        error_rate = final_errors / len(vibe_df)
        print(f"Final error rate after retries: {error_rate}")
        wandb.summary["prop_position_collisions"] = error_rate

        vibe_df["score_label"] = vibe_df["ranker_output"]
        vibe_df["vibe"] = [vibe_batch for _ in range(len(vibe_df))]
        # explode the vibes column and the score column at the same time
        vibe_df = vibe_df.explode(["vibe", "score_label"])
        vibe_df["score"] = vibe_df["score_label"].apply(
            lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
        )
        return vibe_df

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


from components.utils_llm import get_llm_embedding
from sklearn.metrics.pairwise import cosine_similarity
class VibeRankerEmbedding(VibeRanker):
    """
    Uses embedding similarity to score each response pair.
    """
    def __init__(self, config: OmegaConf):
        super().__init__(config)
        self.embedding_model = config["ranker"]["embedding_model"]

    def score_vibe(self, vibe: str, df: pd.DataFrame, models: List[str], single_position_rank: bool = None) -> tuple[pd.DataFrame, float]:
        """
        Scores a single vibe using embedding similarity.
        Returns tuple of (scored_dataframe, num_collisions=0) to match parent class interface.
        """
        
        df = df.copy()
        template = "Rate how much the response has the property '{vibe}'"
        df["vibe_embedding"] = get_llm_embedding(df["vibe"].apply(lambda x: template.format(vibe=x)).tolist(), self.embedding_model)
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
        delta = 0.05
        df["score"] = df.apply(
            lambda row: 1 if row["model_a_vibe_sim"] > row["model_b_vibe_sim"] + delta else (-1 if row["model_a_vibe_sim"] < row["model_b_vibe_sim"] - delta else 0),
            axis=1,
        )

        # Drop embedding columns
        display_df = df.copy()
        display_df = display_df.drop(columns=["model_a_embedding", "model_b_embedding", "vibe_embedding"])

        display_df["score_pos_model"] = [models for _ in range(len(df))] # hack: need this for training prediction models
        # train_embedding_model(df, models, vibe, self.embedding_model, self.config)
        display_df["vibe"] = vibe
        return display_df, 0  
    
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

def train_embedding_model(df: pd.DataFrame, 
                          models: List[str], 
                          vibe: str, 
                          embedding_model: 
                          str, config: OmegaConf) -> None:
    """
    Trains an embedding model on the given data using PyTorch. Ranks half the data using the LLM ranker,
    trains a neural network to predict the ranker score, then uses it to predict the other half.
    This is still in the works.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    df = df.copy()
    config["ranker"]["single_position_rank"] = False
    ranker = VibeRanker(config)
    df, _ = ranker.score_single_vibe(vibe, df, models, single_position_rank=True)
    
    # Get embeddings
    # if "model_a_embedding" not in df.columns:
    #     df["model_a_embedding"] = get_llm_embedding(df[models[0]].tolist(), embedding_model)
    # if "model_b_embedding" not in df.columns:
    #     df["model_b_embedding"] = get_llm_embedding(df[models[1]].tolist(), embedding_model)
    
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
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
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
    # Create plotly bar plot of prediction distribution
    fig = go.Figure(data=[
        go.Bar(
            x=list(df_test["score_pred"].value_counts().index),
            y=list(df_test["score_pred"].value_counts().values)
        )
    ])
    fig.update_layout(
        title=f"Vibe Predictor Distribution - {vibe}",
        xaxis_title="Predicted Score",
        yaxis_title="Count"
    )
    wandb.log({f"Vibe Training/vibe_predictor_pred_dist_{vibe}": fig})
    
    # Calculate and log accuracy
    accuracy = (df_test["score"] == df_test["score_pred"]).mean()
    print(f"accuracy: {accuracy}")
    wandb.log({f"vibe_predictor_accuracy_{vibe}": accuracy})
    
    return df_test
