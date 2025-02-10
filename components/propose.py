import pandas as pd
from typing import List, Tuple
import wandb
import litellm
import numpy as np

from components.prompts.proposer_prompts import proposer_prompt_freeform, proposer_prompt_freeform_iteration


def parse_bullets(text: str):
    lines = text.split("\n")
    bullets = []
    for line in lines:
        if line.strip().startswith("-") or line.strip().startswith("*"):
            bullets.append(line.strip().lstrip("- *").strip())
        else:
            bullets.append(line.strip())
    return bullets


def create_reduce_prompt(num_reduced_axes: int = 0):
    if num_reduced_axes == 0:
        return f"""Below is a list of properties that are found in LLM outputs. I would like to summarize this list to a set of representative properties with clear and concise descriptions that cover the recurring themes in the data. Are there any interesting overarching properties that are present in a large number of the properties? Please return a list of properties that are seen in the data, where each property represents one type of behavior that is seen in the data.

Here is the list of properties:
{{differences}}

A human should be able to understand the property and its meaning, and this property should provide insight into the model's behavior or personality. Do not include subjective analysis about these properties, simply describe the property. For instance "the model is more advanced in its understanding" and "the model uses historical context" is not a good property because it is too vague and does not provide interesting insight into the model's behavior. Similarly, these properties should be on a per prompt basis, so "the model provides a consistent tone across prompts" or "the model varies its tone from formal to informal" is not a good property because a person could not make a judgement only looking at a single prompt. These properties should be something that a human could reasonably expect to see in the model's output when given new prompts. Order your final list of properties by how much they are seen in the data. Your response should be a list deliniated with "-"
"""

    return f"""Below is a list of properties that are found in LLM outputs. I would like to summarize this list to AT MOST {num_reduced_axes} representative properties with clear and concise descriptions. Are there any interesting overarching properties that are present in a large number of the properties?

Here is the list of properties:
{{differences}}

A human should be able to understand the property and its meaning, and this property should provide insight into the model's behavior or personality. Do not include subjective analysis about these properties, simply describe the property. For instance "the model is more advanced in its understanding" or "the model uses historical context" is not a good property because it is too vague and does not provide interesting insight into the model's behavior. These properties should be something that a human could reasonably expect to see in the model's output when given new prompts. Order your final list of properties by how much they are seen in the data. Your response should be a list deliniated with "-"
"""

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

        if current_vibes:
            batch_df = batch_df.sem_map(
                proposer_prompt_freeform_iteration,
                return_raw_outputs=True,
                suffix="differences",
            )
        else:
            batch_df = batch_df.sem_map(
                proposer_prompt_freeform, return_raw_outputs=True, suffix="differences"
            )

        batch_df["differences"] = batch_df["differences"].apply(
            lambda x: [b.replace("**", "").replace("-", "") for b in parse_bullets(x)]
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
        if current_vibes:
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
        # TODO: fix groupby_clusterid for sem_agg
        results = results.sem_index("differences", "differences_index").sem_cluster_by(
            "differences", 1
        )
        print(f"Number of total differences before reduction: {len(results)}")
        summaries = results.sem_agg(
            create_reduce_prompt(0),
            suffix="reduced axes",
        )
        summaries["reduced axes parsed"] = summaries["reduced axes"].apply(parse_axes)
        vibes = summaries.explode("reduced axes parsed")["reduced axes parsed"].to_list()
        print(f"Number of total differences after reduction: {len(vibes)}")
        return vibes[:num_vibes]

# def propose_vibes(
#     df: pd.DataFrame,
#     models: List[str],
#     num_proposal_samples: int = 30,
#     num_final_vibes: int = 10,
#     batch_size: int = 5,
#     current_vibes: List[str] = [],
#     shuffle_positions: bool = False,
# ):
#     df["single_combined_response"] = df.apply(
#         lambda row: (
#             f"User prompt:\n{row['question']}\n\n"
#             f"Model 1:\n{row[models[0]]}\n\n"
#             f"Model 2:\n{row[models[1]]}"
#         ),
#         axis=1,
#     )
#     other_df = df.copy()
#     other_df["single_combined_response"] = other_df.apply(
#         lambda row: (
#             f"User prompt:\n{row['question']}\n\n"
#             f"Model 1:\n{row[models[1]]}\n\n"
#             f"Model 2:\n{row[models[0]]}"
#         ),
#         axis=1,
#     )
#     proposer_df = pd.concat([df, other_df])
#     proposer_df = proposer_df.sample(num_proposal_samples, random_state=42).reset_index(
#         drop=True
#     )
#     proposer_df["batch_id"] = proposer_df.index // batch_size
    
#     # Add shuffling of positions across batches
#     if shuffle_positions:
#         # Randomly decide which batches should have swapped positions
#         swap_batches = np.random.choice([True, False], size=proposer_df["batch_id"].max() + 1)
#         proposer_df["should_swap"] = proposer_df["batch_id"].map(dict(enumerate(swap_batches)))
#         # Apply the swap for selected batches
#         proposer_df.loc[proposer_df["should_swap"], "single_combined_response"] = proposer_df[proposer_df["should_swap"]].apply(
#             lambda row: (
#                 f"User prompt:\n{row['question']}\n\n"
#                 f"Model 1:\n{row[models[1]]}\n\n"
#                 f"Model 2:\n{row[models[0]]}"
#             ),
#             axis=1,
#         )
#         proposer_df = proposer_df.drop("should_swap", axis=1)

#     proposer_df["combined_responses"] = proposer_df.groupby("batch_id")[
#         "single_combined_response"
#     ].transform(lambda x: "\n-------------\n".join(x))
    
#     proposer_df = proposer_df.drop_duplicates("batch_id")
#     if current_vibes:
#         # add current vibes to the combined responses
#         current_vibes_str = "Differences I have already found:\n" + "\n".join(
#             current_vibes
#         )
#         proposer_df["combined_responses"] = proposer_df["combined_responses"].apply(
#             lambda x: x + "\n\n" + current_vibes_str
#         )
#         proposer_df = proposer_df.sem_map(
#             proposer_prompt_freeform_iteration,
#             return_raw_outputs=True,
#             suffix="differences",
#         )
#     else:
#         proposer_df = proposer_df.sem_map(
#             proposer_prompt_freeform, return_raw_outputs=True, suffix="differences"
#         )
 
#     proposer_df["differences"] = proposer_df["differences"].apply(lambda x: [b.replace("**", "").replace("-", "") for b in parse_bullets(x)])
#     results = proposer_df[proposer_df["differences"].apply(lambda x: len(x) > 0)]
#     results = results.explode("differences").reset_index(drop=True)

#     # Cluster and reduce axes
#     # TODO: fix groupby_clusterid for sem_agg
#     results = results.sem_index("differences", "differences_index").sem_cluster_by(
#         "differences", 1
#     )
#     summaries = results.sem_agg(
#         create_reduce_prompt(num_final_vibes),
#         suffix="reduced axes",
#     )
#     summaries["reduced axes parsed"] = summaries["reduced axes"].apply(parse_axes)
#     vibes = summaries.explode("reduced axes parsed")["reduced axes parsed"].to_list()
#     print("Vibes:\n" + "\n".join(vibes))

#     wandb.log({"Vibe Proposer/proposer_results": wandb.Table(dataframe=proposer_df)})
#     return vibes