import pandas as pd
from typing import List, Tuple
import wandb
import litellm
import numpy as np

from components.utils_llm import get_llm_output, get_llm_embedding
import components.prompts.reduction_prompts as reduction_prompts
import components.prompts.proposer_prompts as proposer_prompts


def parse_bullets(text: str):
    lines = text.split("\n")
    bullets = []
    for line in lines:
        if line.strip().startswith("-") or line.strip().startswith("*"):
            bullets.append(line.strip().lstrip("- *").strip())
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

class VibeProposerBase:
    """
    Propose new vibe axes (behaviors).
    """
    from omegaconf import OmegaConf
    def __init__(self, models: List[str], config: OmegaConf):
        self.models = models
        self.config = config
        
    def propose(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Given a dataframe of questions and responses, propose new vibe axes (behaviors).
        """
        return [], df
    
    def propose_batch(self, df: pd.DataFrame, batch_id: int) -> Tuple[List[str], pd.DataFrame]:
        """
        Given a dataframe of questions and responses, propose new vibe axes (behaviors) for a given batch.
        This should return a list of vibe axes.
        """
        pass
    

class VibeProposer(VibeProposerBase):
    """
    An VibeProposer that manages loading, batching, and LLM calls in a 
    structured, step-by-step way. The `propose` method handles dataset preparation
    and batching, while the `propose_batch` method does the actual LLM call(s) 
    and parsing for each batch. This design allows for a clean separation of
    data manipulation and LLM interaction.
    """

    def propose(
        self,
        df: pd.DataFrame,
        current_vibes: List[str] = [],
        **kwargs
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Handle dataset preparation and invoke `propose_batch` for each batch.

        Args:
            df (pd.DataFrame): The DataFrame containing user questions & responses.
            current_vibes (List[str]): Existing vibe axes.
            **kwargs: Override any config values from proposer section:
                - num_proposal_samples (int): Number of samples to randomly select
                - batch_size (int): Size of each batch
                - shuffle_positions (bool): Whether to randomly swap model positions
                - num_vibes (int): Number of vibes to return

        Returns:
            Tuple[List[str], pd.DataFrame]
        """
        # Get configs from self.config and update with any provided kwargs
        configs = {
            'num_proposal_samples': self.config["proposer"].num_samples,
            'batch_size': self.config["proposer"].batch_size,
            'shuffle_positions': self.config["proposer"].shuffle_positions,
            'num_vibes': self.config.num_vibes,
        }
        configs.update(kwargs)

        proposer_df = df.sample(configs['num_proposal_samples'], random_state=42).reset_index(drop=True)

        proposer_df["batch_id"] = proposer_df.index // configs['batch_size']
        unique_batches = proposer_df["batch_id"].unique()

        batch_df = []
        for batch_id in unique_batches:
            batch_df.append(self.prepare_batch(proposer_df[proposer_df["batch_id"] == batch_id].copy(), current_vibes, configs['shuffle_positions']))
        batch_df = pd.concat(batch_df)

        if len(current_vibes) > 0:
            batch_df["differences"] = get_llm_output(
                [getattr(proposer_prompts, self.config["proposer"].iteration_prompt).format(combined_responses=row["combined_responses"])
                 for _, row in batch_df.iterrows()],
                self.config["proposer"].model
            )
        else:
            batch_df["differences"] = get_llm_output(
                [getattr(proposer_prompts, self.config["proposer"].prompt).format(combined_responses=row["combined_responses"])
                 for _, row in batch_df.iterrows()],
                self.config["proposer"].model
            )

        batch_df["differences"] = batch_df["differences"].apply(
            lambda x: [b.replace("**", "") for b in parse_bullets(x)]
        )
        results = batch_df[batch_df["differences"].apply(lambda x: len(x) > 0)]
        results = results.explode("differences").reset_index(drop=True)

        vibes = self.reduce_vibes(results, configs['num_vibes'])

        wandb.log({"Vibe Proposer/proposer_results": wandb.Table(dataframe=results)})
        return vibes, results
    
    def prepare_batch(self, batch_df: pd.DataFrame, current_vibes: List[str] = [], shuffle_positions: bool = False) -> pd.DataFrame:
    
        if shuffle_positions:
            should_swap = np.random.default_rng().choice([True, False])
            if should_swap:
                batch_df["single_combined_response"] = batch_df.apply(
                    lambda row: (
                        f"User prompt:\n{row['question']}\n\n"
                        f"Model 1:\n{row[self.models[1]]}\n\n"
                        f"Model 2:\n{row[self.models[0]]}"
                    ),
                    axis=1,
                )
                # Once swapped, re-generate the combined text
                batch_df["combined_responses"] = "\n-------------\n".join(
                    batch_df["single_combined_response"].tolist()
                )
        else:
            batch_df["single_combined_response"] = batch_df.apply(
                lambda row: (
                    f"User prompt:\n{row['question']}\n\n"
                    f"Model 1:\n{row[self.models[0]]}\n\n"
                    f"Model 2:\n{row[self.models[1]]}"
                ),
                axis=1,
            )
            batch_df["combined_responses"] = "\n-------------\n".join(
                batch_df["single_combined_response"].tolist()
            )   
        if len(current_vibes) > 0:
            current_vibes_str = "Differences I have already found:\n" + "\n".join(current_vibes)
            batch_df["combined_responses"] = batch_df["combined_responses"].apply(
                lambda x: x + "\n\n" + current_vibes_str
            )
        return batch_df.drop_duplicates("batch_id")

    def reduce_vibes(self, results: pd.DataFrame, num_vibes: int) -> List[str]:
        """
        Reduce the list of vibes to a smaller list of vibes.
        """
        # Cluster and reduce axes
        results = results.sem_index("differences", "differences_index").sem_cluster_by(
            "differences", 1
        )
        print(f"Number of total differences before reduction: {len(results)}")
        # sample 100 differences which represent different clusters
        # TODO: change this to k-means clustering (but probably doesn't matter)
        results = results.sample(min(100, len(results)), random_state=42)
        summaries = get_llm_output(getattr(reduction_prompts, self.config["proposer"].reduction_prompt).format(differences='\n'.join(results["differences"].tolist())),
            self.config["proposer"].model
        )

        vibes = parse_axes(summaries)
        print(f"Number of total differences after reduction: {len(vibes)}")
        return vibes[:num_vibes]
