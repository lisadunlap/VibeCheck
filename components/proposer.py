import random
from typing import Dict, List, Tuple

import pandas as pd
from PIL import PngImagePlugin

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

from serve.utils_llm import get_llm_output
from tqdm import tqdm
import omegaconf

import components.proposer_prompts as proposer_prompts
from components.parsing_utils import *


class Proposer:
    def __init__(self, args: Dict):
        self.args = args
        # load default config from yaml configs/base.yaml
        default_args = omegaconf.OmegaConf.load("configs/base.yaml")
        self.args = omegaconf.OmegaConf.merge(default_args, self.args)

    def propose(
        self, dataset1: List[Dict], dataset2: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        all_hypotheses = []
        all_logs = []
        all_images = []
        random.seed(self.args["seed"])
        for i in range(self.args["num_rounds"]):
            sampled_dataset1 = self.sample(dataset1, self.args["num_samples"])
            sampled_dataset2 = self.sample(dataset2, self.args["num_samples"])
            hypotheses, logs = self.get_hypotheses(sampled_dataset1, sampled_dataset2)
            images = self.visualize(sampled_dataset1, sampled_dataset2)
            all_hypotheses += hypotheses
            all_logs.append(logs)
            all_images.append(images)
        return all_hypotheses, all_logs, all_images

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        raise NotImplementedError

    def sample(self, dataset: List[Dict], n: int) -> List[Dict]:
        if self.args["sampling_method"] == "random":
            return random.sample(dataset, n)


class LLMProposer(Proposer):

    question_diff_prompt = """I have a list of user questions group into either A or B and I would like to understand the differences between these groups. Please list any noticeable differences in these groups. please output a list differences between the two outputs with relation to specific axes of variation. Are there any general patterns, clusters, or variations you notice in the outputs? Try to give patterns that are specific enough that someone could reliably produce new questions that would belong to group A or group B, and that they could understand what it means to be higher or lower on that specific axis. 
    
    Here are the questions:
    {text}
    
    Please output a numbered list of differences between the two groups of questions. If there are no clear differences, please output "No differences found"."""

    combine_two_sides = """
    I have two lists of questions, 1 and 2, and I would like to understand the differences between these two groups. To do this I have fed in the questions from both groups into a language model and asked for the differences between the two groups. Here is the output of comparing group 1 and 2 (named A and B):
    
    {left_output}

    To ensure that the differences are not due to the order of the questions, I have also compared group 2 and 1 (group 2 is now A and group 1 is now B). Here is the output of comparing group 2 and 1:

    {right_output}

    Please use this to determine if there are any differences between the two groups of questions that are consistent across both comparisons. For instance, if group 1 was given quality 1 and group 2 quality 2 when comaring groups 1 and 2, this would be correct if when comparing group 2 to group 1 the output gives group 2 quality 1 and group 1 quality 2. If none of the differences are consistent across both comparisons, please output "No consistent differences found".
    """

    def __init__(self, args: Dict):
        super().__init__(args)
        self.batch_size = self.args.batch_size

    def propose_one_side(
        self, texts1: List[str], texts2: List[str]
    ) -> Tuple[List[str], Dict]:
        # batch the texts and call llm to get differences
        prompt = self.question_diff_prompt.format(
            text="Group A:" + "\n".join(texts1) + "\n\nGroup B:" + "\n".join(texts2)
        )
        output = get_llm_output(prompt, model=self.args.proposer_model)
        # converted = get_llm_output(self.conversion.format(axes=output), 'claude-3-opus-20240229')
        logs = {
            "prompt": prompt,
            "output": output,
            "conversion_prompt": self.conversion.format(axes=output),
        }
        return output, logs

    def propose(self, texts1: List[str], texts2: List[str]):
        max_size = 30
        sample_texts_1, sample_texts_2 = random.sample(
            texts1, min(len(texts1), max_size)
        ), random.sample(texts2, min(len(texts2), max_size))
        left_output, left_logs = self.propose_one_side(sample_texts_1, sample_texts_2)
        right_output, right_logs = self.propose_one_side(sample_texts_2, sample_texts_1)

        combined = get_llm_output(
            self.combine_two_sides.format(
                left_output=left_output, right_output=right_output
            ),
            "claude-3-opus-20240229",
        )
        return {
            "left_output": left_output,
            "right_output": right_output,
            "combined": combined,
            "logs": {"left": left_logs, "right": right_logs, "combined": combined},
        }


def extract_questions(text):
    # Remove leading/trailing whitespace and newlines
    text = text.strip()

    # Split the text into lines
    lines = text.split("\n")

    questions = []
    current_question = ""

    for line in lines:
        # Check if the line starts with a number or a bullet point
        if re.match(r"^(\d+|[-*])\.\s", line.strip()):
            if current_question:
                questions.append(current_question.strip())
            current_question = line.strip()
        else:
            current_question += " " + line.strip()

    # Append the last question
    if current_question:
        questions.append(current_question.strip())

    return questions


def parse_bullets(text):
    # Use regex to extract bullet sections, supporting "-", "*", numerical bullets, and others
    bullet_sections = re.split(r"\n\s*-\s*", text.strip())
    print(bullet_sections)
    if bullet_sections[0] == "":
        return []
    print("-----------")

    result = []
    reslts_str = []  # string comprised of category and details

    for section in bullet_sections:
        # Normalize section by removing leading markers and spaces
        section = re.sub(r"^\s*[-*\d.]+", "", section).strip()

        # Split each section based on High/Low points using regular expressions
        title, *details = section.splitlines()
        parsed_details = {}

        for line in details:
            match = re.match(r"\s*(High|Low):\s*(.+)", line)
            if match:
                key, value = match.groups()
                parsed_details[key] = value

        result.append({"Category": title.strip(": \n"), "Details": parsed_details})
        reslts_str.append(title + " " + str(parsed_details))

    return [r.replace("{", "").replace("}", "") for r in reslts_str]


class LLMProposerFixed(Proposer):

    def __init__(self, args: Dict):
        super().__init__(args)
        self.systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes. Your outputs adhere to the format given by the user."
        self.smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."
        self.model_columns = args.models
        self.batch_size = args.proposer_batch_size

    def propose_batch(self, df):
        """
        Get differences over a list of prompts
        """
        axis_convert = """The following are the axes of variation that you can consider when comparing the two outputs along with a description of how language model outputs vary along that axis:

{axes}

I want to formalize exactly what it means to be high and low on each axis. For each axis, I want you to provide a description of what it means to be high and low on that axis so that I can place future model outputs along this axis. If an axis applies to a specific type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to (e.g. code complexity). Your output should be in this format:

- {{axis_1}}:
    High: {{description of high}}
    Low: {{description of low}}

- {{axis_2}}:
    High: {{description of high}}
    Low: {{description of low}}

Please ensure that the description what is high and low on the axis are distinct and mutually exclusive such that given any unseen pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. Please keep the axis name to 10 words or less and descriptions of what is high and low to 10 words or less. If no differences are found, please respond with "No differences found."
"""

        assert "question" in df.columns, "'question' column not in dataset"
        random.seed(self.args.seed)
        # get per question differences
        texts = [
            "Here is a set of user prompts and the responses from two different language models. Rember to find axes of variation in the responses and provide a description of how the responses vary along that axis. Do not answer the user prompts themselves."
        ]
        # shuffle args.models
        # shuffled_cols = random.sample(self.model_columns, len(self.model_columns))
        shuffled_cols = self.model_columns
        for i, row in df.iterrows():
            if not self.args.exclude_question_in_proposer:
                texts.append(f"User prompt:\n{row['question']}")
            for j, model in enumerate(shuffled_cols):
                texts.append(f"\nModel {j}:\n{row[model]}\n")
        texts = "\n".join(texts)
        # prompt = getattr(proposer_prompts, self.args.proposer_prompt).format(text=texts)
        systems_prompt = getattr(proposer_prompts, self.args.proposer_prompt)
        response = get_llm_output(
            texts, model=self.args.proposer_model, system_prompt=systems_prompt
        ).replace("**", "")
        axis_prompt = axis_convert.format(axes=response)
        axis_response = get_llm_output(
            axis_prompt,
            model=self.args.proposer_model,
            system_prompt=self.smaller_systems_prompt,
        )
        return (
            response,
            axis_response,
            {
                "proposal_prompt": self.args.proposer_prompt,
                "response": response,
                "conversion_prompt": axis_prompt,
                "axis_response": axis_response,
            },
        )

    def propose(self, df) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        assert "question" in df.columns, "'question' column not in dataset"
        random.seed(self.args.seed)

        # get per question differences
        results = {
            "question": [],
            self.model_columns[0]: [],
            self.model_columns[1]: [],
            "response": [],
            "axis_response": [],
            "topic": [],
        }
        llm_logs = []
        # partition df by column topic then batch
        topic_dfs = [df[df["topic"] == topic] for topic in df["topic"].unique()]
        for topic_df in topic_dfs:
            print(
                f"Proposing for topic {topic_df['topic'].iloc[0]} of length {len(topic_df)}"
            )
            # add tqdm
            for batch_start in tqdm(range(0, len(topic_df), self.batch_size)):
                batch = topic_df.iloc[batch_start : batch_start + self.batch_size]
                assert batch["topic"].nunique() == 1, "Batch should have the same topic"
                response, axis_response, logs = self.propose_batch(batch)
                results["question"].extend(batch["question"].tolist())
                results[self.model_columns[0]].extend(
                    batch[self.model_columns[0]].tolist()
                )
                results[self.model_columns[1]].extend(
                    batch[self.model_columns[1]].tolist()
                )
                results["response"].extend([response] * len(batch))
                results["axis_response"].extend([axis_response] * len(batch))
                results["topic"].extend(batch["topic"].tolist())
                llm_logs.append(logs)

        results = pd.DataFrame(results)
        pairwise_differences = results[["question", "response", "axis_response"]]
        llm_logs = pd.DataFrame(llm_logs)

        results["no_difference_detected"] = results["response"].apply(
            lambda x: is_match(x, "No differences found")
        )
        results = results[~results["no_difference_detected"]]

        # cluster per axis differences
        results["axis_description"] = results["axis_response"].apply(parse_bullets)
        # remove any empty axis descriptions
        results = results[results["axis_description"].apply(lambda x: len(x) > 0)]
        results = results.explode("axis_description")

        all_axis_descriptions = list(set(results["axis_description"]))
        return all_axis_descriptions, llm_logs, pairwise_differences, results


class LLMProposerIteration(LLMProposerFixed):
    """
    Proposes new axes (vibes) given the existing axes and the misclassified responses
    """

    def __init__(self, args: Dict, axes: List[str]):
        super().__init__(args)
        self.axes = axes

    @staticmethod
    def extract_axes(text):
        # Define regex pattern to match axes and their high/low descriptions
        pattern = r"- ([^\n]+):\n\s+High:(.*?)\n\s+Low:(.*?)\n"
        # Find all matches
        matches = re.findall(pattern, text, re.DOTALL)

        # Format output with just the axes and descriptions
        extracted_axes = ""
        for match in matches:
            axis_name, high_desc, low_desc = match
            extracted_axes += f"- {axis_name.strip()}:\n    High:{high_desc.strip()}\n    Low:{low_desc.strip()}\n\n"

        return extracted_axes.strip()

    def propose_batch(self, df):
        """
        Get differences over a list of prompts
        """
        systems_prompt = """You are an AI researcher looking to compare the behavior of two different LLMs (1 and 2) to determine the defining characteristics of each model. To do this, someone examines a set of responces from 1 and 2 given the same set of questions and asked to find axes in which these models differ. Using these axes, each response pair is ranked as being higher or lower on the axis and these features are used to train a model to predict the model based on where the response falls on each axis.

Given a new set of respenses, your task is to expand on the set of axes which have been previously identified by finding other clear differences between the responses that are not captured by the existing axes. The expanded axes should be any differences between responses that are not clearly captured by the existing axes. Be as exhaustive as possible in listing differences on as many different axes as you can think of, and be specific about what constitutes high and low on each axis. 

Your axis should be interpretable: a human should easily and reliably determine which response is higher, lower, or even on this axis when given a new set of responses. Please do not make your axes too broad and list as many axes as you can think of that are not covered by the existing axes. Most of these new axes should be either completely different from the existing axes or should highlight a more finegrained difference which an existing axis might broadly cover. For instance, if an existing axis is "Enthusiasm: High: enthusiastic, Low: unenthusiastic", a new axis might be "Use of Exclamation Points", or if an existing axis is "Cultural Context: High: culturally relevant, Low: culturally irrelevant", a new axis might be "Use of Slang". a new axis might be "Use of Exclamation Points", or if an existing axis is "Context", a new axis might be "".

Please think through the axes carefully and make sure they are clear, concise, and do not overlap with eachother or the existing axes. Do not include any of the existing axes in your response. Your output should be in this format:

New Axes:
- {{axis_1}}:
    High: {{description of high}}
    Low: {{description of low}}

- {{axis_2}}:
    High: {{description of high}}
    Low: {{description of low}}

Do not include any other information in your response.
"""

        assert "question" in df.columns, "'question' column not in dataset"
        random.seed(self.args.seed)
        print("PROPOSER ITERATION")

        # get per question differences
        texts = []
        # shuffle args.models
        shuffled_cols = self.model_columns
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            if not self.args.exclude_question_in_proposer:
                texts.append(f"Question:\n{row['question']}")
            for j, model in enumerate(shuffled_cols):
                texts.append(f"\Model {j}:\n{row[model]}\n")
        texts = (
            "Current Axes:\n"
            + "\n".join(self.axes)
            + "\n\nMisclassified Responses:"
            + "\n".join(texts)
        )
        response = get_llm_output(
            texts, model=self.args.proposer_model, system_prompt=systems_prompt
        ).replace("**", "")
        axis_response = self.extract_axes(response)
        print(axis_response)
        return (
            response,
            axis_response,
            {
                "proposal_prompt": self.args.proposer_prompt,
                "response": response,
                "axis_response": axis_response,
            },
        )


class DummyProposer(Proposer):
    """
    Proposes possible ways an LLM output can differ from another LLM output without actually showing model outputs
    """

    def __init__(self, args: Dict):
        super().__init__(args)

    def get_hypotheses(self, df):
        prompt = """I am a machine learning researcher trying to figure out the major differences between the behavior of different large language mdoels. Can you list common ways in which two language models can differ in their outputs?
        
        Please output a list differences between these sets of outputs with relation to specific axes of variation. Try to give axes that a human could easily interpret and they could understand what it means to be higher or lower on that specific axis. Please ensure that the concepts used to explain what is high and low on the axis are distinct and mutually exclusive such that given any tuple of text outputs, a human could easily and reliably determine which model is higher or lower on that axis.
        
        The format should be
        - {{axis_1}}: {{difference}}
        - {{axis_2}}: {{difference}}
            
        Please output differences which have a possibility of showing up in future unseen data and which would be useful for a human to know about when deciding with LLM to use. For each axis, define clearly and succinctly what constitutes a high or low score, ensuring these definitions are mutually exclusive."""

        axis_convert = """The following are the axes of variation that you can consider when comparing the two outputs along with a description of how language model outputs vary along that axis:

            {axes}

            I want to formalize exactly what it means to be high and low on each axis. For each axis, I want you to provide a description of what it means to be high and low on that axis so that I can place future model outputs along this axis. Your output should be in this format:

            - {{axis_1}}:
                High: {{description of high}}
                Low: {{description of low}}

            - {{axis_2}}:
                High: {{description of high}}
                Low: {{description of low}}

            Please ensure that the description what is high and low on the axis are distinct and mutually exclusive such that given any unseen pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. Please keep the axis name and descriptions of what is high and low are less than 10 words each.
        """

        response = get_llm_output(
            prompt,
            model=self.args.proposer_model,
            system_prompt="You are a helpful assistant. Your outputs adhere to the format given by the user.",
        )
        axis_response = get_llm_output(
            axis_convert.format(axes=response),
            model=self.args.proposer_model,
            system_prompt="You are a helpful assistant. Your outputs adhere to the format given by the user.",
        )

        return (
            response,
            axis_response,
            {
                "proposal_prompt": prompt,
                "response": response,
                "axis_response": axis_response,
            },
        )

    def propose(self, df) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        assert "question" in df.columns, "'question' column not in dataset"
        random.seed(self.args.seed)

        # get per question differences
        results = {"question": [], "response": [], "axis_response": [], "topic": []}
        llm_logs = []
        response, axis_response, logs = self.propose_batch(df)
        results["question"].extend(df["question"].tolist())
        results["response"].extend([response] * len(df))
        results["axis_response"].extend([axis_response] * len(df))
        results["topic"].extend(df["topic"].tolist())
        llm_logs.append(logs)

        results = pd.DataFrame(results)
        pairwise_differences = results[["question", "response", "axis_response"]]
        llm_logs = pd.DataFrame(llm_logs)

        # cluster per axis differences
        results["axis_description"] = results["axis_response"].apply(parse_bullets)
        results = results.explode("axis_description")

        all_axis_descriptions = list(set(results["axis_description"]))
        return all_axis_descriptions, llm_logs, pairwise_differences, results
