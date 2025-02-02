import pandas as pd
from typing import List
import wandb
import litellm


def parse_bullets(text: str):
    """
    Parse bullet points from text.
    """
    lines = text.split("\n")
    bullets = []
    for line in lines:
        if line.strip().startswith("-") or line.strip().startswith("*"):
            bullets.append(line.strip().lstrip("- *").strip())
    return bullets


def create_reduce_prompt(num_reduced_axes: int):
    return f"""Below is a list of properties that are found in LLM outputs. I would like to summarize this list to AT MOST {num_reduced_axes} representative properties with clear and concise descriptions. Are there any interesting overarching properties that are present in a large number of the properties?

Here is the list of properties:
{{differences}}

A human should be able to understand the property and its meaning, and this property should provide insight into the model's behavior or personality. Do not include subjective analysis about these properties, simply describe the property. For isntance "the model is more advanced in its understanding" is not a good property because it is too vague and does not provide interesting insight into the model's behavior. These properties should be something that a human could reasonably expect to see in the model's output when given new prompts. Order your final list of properties by how much they are seen in the data. Your response should be a list deliniated with "-"
"""

def remove_duplicate_vibes(vibes: List[str]):
    prompt = """Here is a list of properties that an LLM output could have. It is likely that several of these properties measure similar things. Your task is to remove any redundant properties. Think about if a user would gain any new information from seeing both properties. For example, "entusiastic tone" and "excited tone" are redundant because they both measure the emotional content of the text. If two similar properties are found, keep the one that is more informative. Here is the list of properties:
{vibes}

Return the reduced list of properties as a list, with each property on a new line."""
    response = pd.DataFrame({"vibes": vibes}).sem_agg(prompt)
    return response._output[0].strip().split("\n")


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


def propose_vibes(
    df: pd.DataFrame,
    models: List[str],
    num_proposal_samples: int = 30,
    num_final_vibes: int = 10,
    batch_size: int = 5,
    current_vibes: List[str] = [],
    shuffle_positions: bool = False,
):
    proposer_prompt_freeform = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. Write down as many differences as you can find between the two outputs. Please format your differences as a list of properties that appear more in one output than the other.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, analyze the differences between Model 1 and Model 2. What properties are seen in the responses from Model 1 that are not seen in the responses from Model 2? What properties are seen in the responses from Model 2 that are not seen in the responses from Model 1?

{combined_responses}

The format should be a list of properties that appear more in one output than the other in the format of a short description of the property. Respond with a list of properties, each on a new line.

Note that this example is not at all exhaustive, but rather just an example of the format. Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic, also consider differences which are relevant to that property or topic.
    
Remember that these differences should be human interpretable and that the differences should be concise (<= 10 words), substantive and objective. Write down as many properties as you can find. Do not explain which model has which property, simply describe the property.
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

    proposer_prompt_freeform_iteration = """You are a machine learning researcher trying to figure out the major differences between the behaviors of two llms by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. I have already found some differences between the two outputs, but there are many more differences to find. Write down as many differences as you can find between the two outputs which are not already in the list of differences. Please format your differences as a list of properties that appear more in one output than the other.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, analyze the differences between Model 1 and Model 2. What properties are seen in the responses from Model 1 that are not seen in the responses from Model 2? What properties are seen in the responses from Model 2 that are not seen in the responses from Model 1? Here are the differences I have already found and the questions and responses:

{combined_responses}

The format should be a list of properties that appear more in one output than the other in the format of a short description of the property. Respond with a list of properties, each on a new line.

Note that this example is not at all exhaustive, but rather just an example of the format. Consider differences on many different axes such as tone, language, structure, content, safety, and any other axis that you can think of. If the questions have a specific property or cover a specific topic, also consider differences which are relevant to that property or topic.
    
Remember that these differences should be human interpretable and that the differences should be concise (<= 10 words), substantive and objective. Write down as many properties as you can find which are not already represented in the list of differences. Do not explain which model has which property, simply describe the property.
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

    # Create combined responses to get in LOTUS format
    df["single_combined_response"] = df.apply(
        lambda row: (
            f"User prompt:\n{row['question']}\n\n"
            f"Model 1:\n{row[models[0]]}\n\n"
            f"Model 2:\n{row[models[1]]}"
        ),
        axis=1,
    )
    other_df = df.copy()
    other_df["single_combined_response"] = other_df.apply(
        lambda row: (
            f"User prompt:\n{row['question']}\n\n"
            f"Model 1:\n{row[models[1]]}\n\n"
            f"Model 2:\n{row[models[0]]}"
        ),
        axis=1,
    )
    proposer_df = pd.concat([df, other_df])
    proposer_df = proposer_df.sample(num_proposal_samples, random_state=42).reset_index(
        drop=True
    )
    proposer_df["batch_id"] = proposer_df.index // batch_size
    proposer_df["combined_responses"] = proposer_df.groupby("batch_id")[
        "single_combined_response"
    ].transform(lambda x: "\n-------------\n".join(x))
    proposer_df = proposer_df.drop_duplicates("batch_id")
    if current_vibes:
        # add current vibes to the combined responses
        current_vibes_str = "Differences I have already found:\n" + "\n".join(
            current_vibes
        )
        proposer_df["combined_responses"] = proposer_df["combined_responses"].apply(
            lambda x: x + "\n\n" + current_vibes_str
        )
        proposer_df = proposer_df.sem_map(
            proposer_prompt_freeform_iteration,
            return_raw_outputs=True,
            suffix="differences",
        )
    else:
        proposer_df = proposer_df.sem_map(
            proposer_prompt_freeform, return_raw_outputs=True, suffix="differences"
        )
 
    proposer_df["differences"] = proposer_df["differences"].apply(lambda x: [b.replace("**", "").replace("-", "") for b in parse_bullets(x)])
    results = proposer_df[proposer_df["differences"].apply(lambda x: len(x) > 0)]
    results = results.explode("differences").reset_index(drop=True)

    # Cluster and reduce axes
    # TODO: fix groupby_clusterid for sem_agg
    results = results.sem_index("differences", "differences_index").sem_cluster_by(
        "differences", 1
    )
    summaries = results.sem_agg(
        create_reduce_prompt(num_final_vibes),
        suffix="reduced axes",
    )
    summaries["reduced axes parsed"] = summaries["reduced axes"].apply(parse_axes)
    vibes = summaries.explode("reduced axes parsed")["reduced axes parsed"].to_list()
    print("Vibes:\n" + "\n".join(vibes))

    wandb.log({"Vibe Proposer/proposer_results": wandb.Table(dataframe=proposer_df)})
    return vibes
