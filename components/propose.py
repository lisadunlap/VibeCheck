import pandas as pd
from typing import List, Tuple
import wandb
import numpy as np

from components.utils_llm import get_llm_output
import components.prompts.reduction_prompts as reduction_prompts
import components.prompts.proposer_prompts as proposer_prompts


def parse_bullets(text: str):
    lines = text.split("\n")
    bullets = []
    for line in lines:
        if line.strip().startswith("-") or line.strip().startswith("*"):
            # oply remove the first * if
            bullets.append(line.replace("* ", "").replace("- ", "").strip())
    return bullets

class VibeProposerBase:
    """
    Propose new vibe axes (behaviors).
    """
    from omegaconf import OmegaConf
    def __init__(self, models: List[str], config: OmegaConf):
        self.models = models
        self.global_config = config
        self.config = config["proposer"]

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
        proposer_df: pd.DataFrame,
        current_vibes: List[str] = [],
        num_vibes: int = 10,
        **kwargs
    ) -> List[str]:
        """
        Given a dataframe of questions and responses, propose new vibe axes (behaviors). 
        If existing vibes are provided, use them to guide the proposal to find new differences.

        Args:
            proposer_df (pd.DataFrame): The DataFrame containing user questions & responses.
            current_vibes (List[str]): Existing vibe axes.
            num_vibes (int): Number of vibes to return.
            **kwargs: Override any config values from proposer section (args can be found in 'configs/base.yaml')
        """
        for key, value in kwargs.items():
            setattr(self.config, key, value)

        proposer_df["batch_id"] = proposer_df.index // self.config.batch_size
        unique_batches = proposer_df["batch_id"].unique()

        differences = []
        for batch_id in unique_batches:
            batch_differences = self.propose_batch(proposer_df[proposer_df["batch_id"] == batch_id].copy(), current_vibes, self.config.shuffle_positions)
            differences.extend(batch_differences)

        vibes = self.reduce_vibes(differences, num_vibes=num_vibes)

        wandb.log({"Vibe Proposer/proposer_results": wandb.Table(dataframe=pd.DataFrame(differences, columns=["differences"]))})
        return vibes
    
    def prepare_batch(self, batch_df: pd.DataFrame, current_vibes: List[str] = [], shuffle_positions: bool = False) -> pd.DataFrame:
        def create_combined_response(row, model_order):
            return (
                f"User prompt:\n{row['question']}\n\n"
                f"Model 1:\n{row[model_order[0]]}\n\n"
                f"Model 2:\n{row[model_order[1]]}"
            )

        model_order = [self.models[0], self.models[1]]
        if shuffle_positions and np.random.default_rng().choice([True, False]):
            model_order.reverse()

        batch_df["single_combined_response"] = batch_df.apply(
            lambda row: create_combined_response(row, model_order),
            axis=1,
        )
        batch_df["combined_responses"] = "\n-------------\n".join(
            batch_df["single_combined_response"].tolist()
        )

        if current_vibes:
            current_vibes_str = "Differences I have already found:\n" + "\n".join(current_vibes)
            batch_df["combined_responses"] = batch_df["combined_responses"].apply(
                lambda x: x + "\n\n" + current_vibes_str
            )

        return batch_df.drop_duplicates("batch_id")

    def propose_batch(self, batch_df: pd.DataFrame, current_vibes: List[str], shuffle_positions: bool) -> List[str]:
        """
        Given a batch DataFrame, propose new vibe axes (behaviors) and return differences.

        Args:
            batch_df (pd.DataFrame): The DataFrame for the current batch.
            current_vibes (List[str]): Existing vibe axes.
            shuffle_positions (bool): Whether to randomly swap model positions.

        Returns:
            List[str]: A list of differences.
        """
        batch_df = self.prepare_batch(batch_df, current_vibes, shuffle_positions)
        proposer_prompt = getattr(proposer_prompts, self.config.prompt) if len(current_vibes) == 0 else getattr(proposer_prompts, self.config.iteration_prompt)
        differences = get_llm_output(
            [proposer_prompt.format(combined_responses=row["combined_responses"])
                for _, row in batch_df.iterrows()],
            self.config.model
        )

        differences = [b.replace("**", "") for diff in differences for b in parse_bullets(diff)]
        return differences

    def reduce_vibes(self, differences: List[str], num_vibes: int) -> List[str]:
        """
        Reduce the list of differences to a smaller list of vibes.
        """
        print(f"Number of total differences before reduction: {len(differences)}")
        summaries = get_llm_output(
            getattr(reduction_prompts, self.config.reduction_prompt).format(differences='\n'.join(differences)),
            self.config.model
        )

        vibes = parse_bullets(summaries)
        vibes = [vibe.replace("*", "") for vibe in vibes]
        print(f"Number of total differences after reduction: {len(vibes)}")
        return vibes[:num_vibes]
