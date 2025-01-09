import random
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, mode
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange
import re
import ast
import itertools
import plotly.graph_objects as go
import plotly.express as px

import wandb
from serve.utils_llm import get_llm_output

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm


class Ranker:
    def __init__(self, args: Dict):
        random.seed(args["seed"])
        self.args = args
        if "group_names" in args:
            self.group_names = args["group_names"]
        else:
            self.group_names = ["Group A", "Group B"]

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        raise NotImplementedError

    def rerank_hypotheses(
        self, hypotheses: List[str], dataset1: List[dict], dataset2: List[dict]
    ) -> List[dict]:
        if len(dataset1) > self.args["max_num_samples"]:
            random.seed(self.args["seed"])
            dataset1 = random.sample(dataset1, self.args["max_num_samples"])
        if len(dataset2) > self.args["max_num_samples"]:
            random.seed(self.args["seed"])
            dataset2 = random.sample(dataset2, self.args["max_num_samples"])

        scored_hypotheses = []
        for hypothesis in tqdm(hypotheses):
            scores1 = self.score_hypothesis(hypothesis, dataset1)
            scores2 = self.score_hypothesis(hypothesis, dataset2)

            metrics = self.compute_metrics(scores1, scores2, hypothesis)
            scored_hypotheses.append(metrics)
        scored_hypotheses = sorted(
            scored_hypotheses, key=lambda x: x["auroc"], reverse=True
        )
        return scored_hypotheses


class NullRanker(Ranker):
    def __init__(self, args: Dict):
        super().__init__(args)

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        return [0.0] * len(dataset)


def aggregate_scores(scores):
    # given a  num_items x num_judges matrix of scores, aggregate the scores into a single score per item
    mode_score, count = mode(scores, axis=0, keepdims=False)
    average_score = np.mean(scores, axis=0)
    # round the average score
    # average_score = np.round(average_score)
    majority_vote = mode_score[0] if mode_score.size > 0 else None  # Handle empty input

    return {
        "Majority Vote": majority_vote,
        "Average Score, Rounded": np.round(average_score),
        "Average Score": average_score,
    }


def top_high_variance_indices(scores, top_n=5):
    """
    Identify indices with the highest variance in scores across judges.
    """
    judges = list(scores.keys())
    scores_matrix = np.array(
        [scores[judge] for judge in judges]
    ).T  # transpose to have judges as columns
    variances = np.var(scores_matrix, axis=1)
    top_variance_indices = np.argsort(-variances)[
        :top_n
    ]  # argsort in descending order by using negative variances

    return np.array(top_variance_indices.tolist())


class RelativeRanker(Ranker):
    """
    Scores by saying which model fits the description better
    """

    def __init__(self, args: Dict):
        super().__init__(args)
        random.seed(args.seed)
        self.num_judges = len(self.args.judges)

    def reformat_response(self, response):
        formatting_prompt = """Given the following LLM reponse, reformat the response to the following format:
        Analysis: {{reasoning}}
        Model: {{A, B, or N/A}}
        
        Reformatted Response:"""
        return get_llm_output(formatting_prompt + response, model="gpt-4o-mini")

    def extract_scores(self, output):
        """parse out the score from the output of the following format
        Analysis: {{reasoning}}
        Model: {{A, B, or N/A}}
        """

        def helper(output):
            # remove any # or * characters
            output = output.replace("Output ", "").replace("output ", "")
            output = re.sub(r"[#*]", "", output)
            # ignor case and multiline
            score_pattern = re.compile(r"Model: (A|B|N/A|unsure|equal)", re.I | re.M)
            score = score_pattern.findall(output)
            if len(score) == 0:
                return -100
            if score[0] == "A" or score[0] == "a":
                return 1
            elif score[0] == "B" or score[0] == "b":
                return -1
            elif score[0] == "N/A" or score[0] == "n/a":
                return 0
            elif score[0] == "unsure" or score[0] == "Unsure":
                print("Unsure")
                return 0
            elif score[0] == "equal" or score[0] == "Equal":
                return 0

        score = helper(output)
        if score == -100:
            output = self.reformat_response(output)
            score = helper(output)
            return score if score != -1 else 0
        return score

    def get_score(self, row, axis, dummy_eval=False):
        if dummy_eval:
            rand = np.random.rand()
            if rand < 0.33:
                return [
                    ["Analysis: Because I said so\nModel: A"] * len(self.args.models)
                    for i in range(self.num_judges)
                ]
            elif rand < 0.66:
                return [
                    ["Analysis: Because I said so\nModel: B"] * len(self.args.models)
                    for i in range(self.num_judges)
                ]
            else:
                return [
                    ["Analysis: Because I said so\nModel: N/A"] * len(self.args.models)
                    for i in range(self.num_judges)
                ]

        prompt = """Axis = {axis}

        {prompt}
        """

#         judge_systems_prompt = """You are a fair and unbiased judge. Your task is to compare the ouputs of two lamgauge models (A and B) on a given axis, which contains a description of what it means for an output to be high and low on that axis. If you had to choose which output is higher on the axis, which would you choose? Please respond with which model response you think is higher on the axis and explain your reasoning. Note that being high or low on the axis does not relate to how good or bad the reponse is, the sole focus is to put an ordering on the responses if they differ on this axis. Avoid any position biase and be as objective as possible. If the response A is higher on the axis, respond with "A", if response B is higher, respond with "B", and if they are roughly equal on this axis or this axis, return "equal". If this axis not apply to these outputs (e.g. the axis is about code quality but the prompt provided is not a coding question), return "N/A". If you are unsure of the meaning of the axis, return "unsure". Use the following format for your response:

# Analysis: {{reasoning}}
# Model: {{A, B, equal, N/A, or unsure}}

# Remember to be as objective as possible and strictly adhere to the response format.
# """

        judge_systems_prompt = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a given axis. Each axis contains a description explaining what it means for an output to be high or low. Your goal is to decide which model’s output is higher on the axis.
When comparing the outputs, consider the following:

	•	Being high or low on the axis does not indicate how good or bad the response is. Your sole focus is to order the responses based on how they differ on this axis.
	•	Avoid any position bias and remain as objective as possible.

Instructions:
	•	If Response A aligns with the 'high' description more than Response B, respond with “A”.
	•	If Response B aligns with the 'high' description more than Response A, respond with “B”.
	•	If the responses are roughly equal on the axis, respond with “equal”.
	•	If the axis does not apply to these outputs (e.g., the axis is about code quality, but the prompt is not related to coding), respond with “N/A”.
	•	If you are unsure about the meaning of the axis, respond with “unsure”.

Use the following format for your response:

Analysis: {{reasoning}}
Model: {{A, B, equal, N/A, or unsure}}

Remember to be as objective as possible and strictly adhere to the response format."""

        judge_outputs = []
        for judge in self.args.judges:
            # print(f"Getting judgement for Judge = {judge}")
            model_outputs = []
            for model in self.args.models:
                model_a, model_b = model, [m for m in self.args.models if m != model][0]
                scoring_prompt = prompt.format(
                    axis=axis,
                    prompt=f"Prompt: {row['question']}\nOutput A: {row[model_a]}\nOutput B: {row[model_b]}",
                )
                output_a = get_llm_output(
                    scoring_prompt, model=judge, system_prompt=judge_systems_prompt
                )
                # score = self.parse_output(output_a)
                model_outputs.append(output_a)
            judge_outputs.append(model_outputs)
        return judge_outputs

    def score_hypothesis(self, hypothesis: str, dataset: List[dict]) -> List[float]:
        from components.mm_and_pp_modeling import get_score

        print(f"Scoring hypothesis {hypothesis}")
        judge_scores = {}
        for i in range(self.num_judges):
            judge_scores[f"Judge_{i}_scores"] = []
            judge_scores[f"Judge_{i}_final_scores"] = []
        judge_scores["avg_scores"] = []
        judge_scores["avg_diff_scores"] = []
        judge_scores["avg_final_scores"] = []
        dataset_scores = []

        def process_row(row):
            scores = self.get_score(row, hypothesis, dummy_eval=self.args.dummy_eval)
            if scores is not None:
                for i, score in enumerate(scores):
                    row[f"Judge_{i}_scores_reasoning"] = score
                    row[f"Judge_{i}_score"] = [self.extract_scores(s) for s in score]
                    row[f"Judge_{i}_diff_score"] = [
                        row[f"Judge_{i}_score"][j] - np.mean(row[f"Judge_{i}_score"])
                        for j in range(len(row[f"Judge_{i}_score"]))
                    ]
                    row[f"Judge_{i}_final_score"] = get_score(row[f"Judge_{i}_score"])
                    row["axis"] = hypothesis
            else:
                print("No scores found")
            row["avg_scores"] = aggregate_scores(
                np.array([row[f"Judge_{i}_score"] for i in range(self.num_judges)])
            )["Average Score"]
            row["avg_diff_scores"] = aggregate_scores(
                np.array([row[f"Judge_{i}_diff_score"] for i in range(self.num_judges)])
            )["Average Score"]
            row["avg_final_scores"] = np.average(
                [row[f"Judge_{i}_final_score"] for i in range(self.num_judges)]
            )
            row["score"] = get_score(row["avg_diff_scores"])
            return row, scores

        with ThreadPoolExecutor(
            max_workers=min(len(dataset), self.args.num_workers)
        ) as executor:
            future_to_row = {executor.submit(process_row, row): row for row in dataset}

            for future in tqdm(as_completed(future_to_row), total=len(dataset)):
                row, scores = future.result()
                if scores is not None:
                    for i in range(self.num_judges):
                        judge_scores[f"Judge_{i}_scores"].append(
                            row[f"Judge_{i}_score"]
                        )
                        judge_scores[f"Judge_{i}_final_scores"].append(
                            row[f"Judge_{i}_final_score"]
                        )
                judge_scores["avg_scores"].append(row["avg_scores"])
                judge_scores["avg_diff_scores"].append(row["avg_diff_scores"])
                judge_scores["avg_final_scores"].append(row["avg_final_scores"])
                dataset_scores.append(row)
        return judge_scores, dataset_scores, {}

    def score(self, axes: List[str], dataset: List[dict]):
        all_dataset_scores, all_logs, axis_metrics = [], [], []
        for axis in axes:
            if self.args.early_stopping:
                # try on 25 rows, if the score differences are < 0.1, continue
                scores, dataset_scores, logs = self.score_hypothesis(axis, dataset[:50])
                dscores = pd.DataFrame(dataset_scores)
                seperability_score = np.mean(dscores["avg_final_scores"])
                if np.abs(seperability_score) < self.args.early_stopping_threshold:
                    print(f"Skipping {axis} (seperability score: {seperability_score})")
                    wandb.summary[f"{axis}_seperability_score"] = seperability_score
                    continue
                if len(self.args.judges) > 1:
                    judge_1_scores = dscores["Judge_0_final_score"].tolist()
                    judge_2_scores = dscores["Judge_1_final_score"].tolist()
                    cohns_kappa = cohen_kappa_score(judge_1_scores, judge_2_scores)
                    if cohns_kappa < 0.2:
                        print(f"Skipping axis {axis} (Cohens kappa: {cohns_kappa})")
                        wandb.summary[f"{axis}_cohn_kappa"] = cohns_kappa
                        continue

            scores, dataset_scores, logs = self.score_hypothesis(axis, dataset)
            all_dataset_scores.append(pd.DataFrame(dataset_scores))
            all_logs.append(logs)
            metrics = self.compute_metrics(axis, scores)
            axis_metrics.append(metrics)
        if len(axis_metrics) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        return (
            pd.DataFrame(axis_metrics),
            pd.concat(all_dataset_scores),
            pd.DataFrame(all_logs),
        )

    def compute_metrics(self, axis, scores, plot=True):
        from sklearn.metrics import cohen_kappa_score
        from itertools import combinations

        metrics = {"axis": axis}
        if plot:
            self.plot_score_distribution(axis, scores, self.args.models)

        for m, model in enumerate(self.args.models):
            judge_pairs = list(
                combinations(range(self.num_judges), 2)
            )  # List of all pairs of judges
            for judge_pair in judge_pairs:
                judge_1, judge_2 = judge_pair
                # Get scores for the two judges
                scores_1 = np.array(scores[f"Judge_{judge_1}_scores"])[:, m]
                scores_2 = np.array(scores[f"Judge_{judge_2}_scores"])[:, m]
                # Calculate Cohen's kappa between the two judges
                kappa = cohen_kappa_score(scores_1, scores_2, weights="linear")
                # Add result to metrics with a unique key for this pair
                metrics[
                    f"{model} Cohen's Kappa (Judge {judge_1} vs Judge {judge_2})"
                ] = kappa
            for judge in range(self.num_judges):
                scores_list = np.array(scores[f"Judge_{judge}_scores"])
                # get the m col of scores list
                scores_model = scores_list[:, m]
                # get average across the models
                model_avg_score = np.average(scores_list, axis=1)
                score_diff = scores_model - model_avg_score

                metrics[f"Judge_{judge}_{model}_mean_score"] = np.round(
                    np.average(scores_model), 3
                )
                metrics[f"Judge_{judge}_{model}_mean_diff"] = np.round(
                    np.mean(score_diff), 3
                )
                metrics[f"Judge_{judge}_{model}_mean_diff_sign"] = np.round(
                    np.mean(np.sign(score_diff)), 3
                )

            # compute stats for majority_score
            if len(scores_model) == 0:
                raise ValueError(f"No scores found for axis {axis}")
            scores_list = np.array(scores["avg_scores"])
            scores_model = scores_list[:, m]
            model_avg_score = np.average(scores_list, axis=1)
            score_diff = scores_model - model_avg_score
            metrics[f"Judge_avg_{model}_mean_score"] = np.round(
                np.average(scores_model), 3
            )
            metrics[f"Judge_avg_{model}_mean_diff"] = np.round(np.mean(score_diff), 3)
            metrics[f"Judge_avg_{model}_mean_diff_sign"] = np.round(
                np.mean(np.sign(score_diff)), 3
            )
            # get normalized value counts of scores
            metrics[f"Judge_avg_{model}_mean_score_counts"] = str(
                {
                    i: np.round(np.sum(scores_model == i) / len(scores_model), 2)
                    for i in range(-2, 3)
                }
            )
            metrics[f"Judge_avg_{model}_mean_diff_counts"] = str(
                {
                    i: np.round(np.sum(score_diff == i) / len(score_diff), 2)
                    for i in range(-5, 6)
                }
            )
            metrics[f"Judge_avg_{model}_mean_diff_sign_counts"] = str(
                {
                    i: np.round(np.sum(np.sign(score_diff) == i) / len(score_diff), 2)
                    for i in range(-1, 2)
                }
            )

        # do a paired t_test for the per-sample scores averaged across judges for each set of models
        model_pairs = list(combinations(self.args.models, 2))
        for model_pair in model_pairs:
            model_1, model_2 = model_pair
            model_idxs = [
                self.args.models.index(model_1),
                self.args.models.index(model_2),
            ]
            scores_1 = np.average(
                np.array(
                    [
                        np.array(scores[f"Judge_{judge}_scores"])[:, model_idxs[0]]
                        for judge in range(self.num_judges)
                    ]
                ),
                axis=0,
            )
            scores_2 = np.average(
                np.array(
                    [
                        np.array(scores[f"Judge_{judge}_scores"])[:, model_idxs[1]]
                        for judge in range(self.num_judges)
                    ]
                ),
                axis=0,
            )
            t_statistic, p_value = ttest_rel(scores_1, scores_2)
            metrics[f"t_statistic_{model_1}_{model_2}"] = t_statistic
            metrics[f"p_value_{model_1}_{model_2}"] = p_value
            t_statistic_sign, p_value_sign = ttest_rel(
                np.sign(scores_1), np.sign(scores_2)
            )
            metrics[f"t_statistic_sign_{model_1}_{model_2}"] = t_statistic_sign
            metrics[f"p_value_sign_{model_1}_{model_2}"] = p_value_sign

        metrics["support"] = len(scores_list)

        return metrics

    @staticmethod
    def plot_score_distribution(axis, scores, models):
        plotting_data = {"model": [], "score": []}
        for m, model in enumerate(models):
            scores_list = np.array(scores["avg_scores"])
            scores_model = scores_list[:, m]
            plotting_data["model"].extend([model] * len(scores_model))
            plotting_data["score"].extend(scores_model)

        # Convert the plotting data to DataFrame
        df = pd.DataFrame(plotting_data)

        # Plot using seaborn's countplot
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x="score", hue="model", palette="viridis")
        ax.set_title(f"{axis}")
        plt.legend(title="Model")

        # Log the plot to W&B
        fig = ax.get_figure()
        wandb.log({f"Juge Scores/{axis.split(':')[0]} Score Counts": wandb.Image(fig)})
        plt.close(fig)

class RelativeRanker2(RelativeRanker):
    """
    Scores by saying which model fits the description better
    """

    def get_score(self, row, axis, dummy_eval=False):
        if dummy_eval:
            rand = np.random.rand()
            if rand < 0.33:
                return [
                    ["Analysis: Because I said so\nModel: A"] * len(self.args.models)
                    for i in range(self.num_judges)
                ]
            elif rand < 0.66:
                return [
                    ["Analysis: Because I said so\nModel: B"] * len(self.args.models)
                    for i in range(self.num_judges)
                ]
            else:
                return [
                    ["Analysis: Because I said so\nModel: N/A"] * len(self.args.models)
                    for i in range(self.num_judges)
                ]

        prompt = """Property = {axis}

{prompt}
"""

        judge_systems_prompt = """You are a fair and unbiased judge. Your task is to compare the outputs of two language models (A and B) on a given propoery. Which repose better aligns more with the given property, A, B, or equal?
When comparing the outputs, consider the following:

	•	Your sole focus is to determine which response better aligns with the given property, NOT how good or bad the response is.
	•	Avoid any position bias and remain as objective as possible.
    •	Consider what the property means and how it applies to the outputs. Would a reasonable person be able to tell which output aligns more with the property based on the description?

Instructions:
	•	If Response A aligns with the property more than Response B, respond with “A”.
    •	If Response B aligns with the property more than Response A, respond with “B”.
	•	If the responses are roughly equal on the property, respond with “equal”.
	•	If the property does not apply to these outputs (e.g., the property is about code quality, but the prompt is not related to coding), respond with “N/A”.
	•	If you are unsure about the meaning of the property, respond with “unsure”. Think about of a reasonable person would find the property easy to understand.

A group of humans should agree with your decision. Use the following format for your response:
Model: {{A, B, equal, N/A, or unsure}}

Remember to be as objective as possible and strictly adhere to the response format."""

        judge_outputs = []
        for judge in self.args.judges:
            # print(f"Getting judgement for Judge = {judge}")
            model_outputs = []
            for model in self.args.models:
                model_a, model_b = model, [m for m in self.args.models if m != model][0]
                if "low" in axis.lower().split("high:")[1].split("low:")[0].replace("", ""):
                    print("Error parsing axis", axis)
                scoring_prompt = prompt.format(
                    axis=axis.lower().split("high:")[1].split("low:")[0].replace("", "").strip(),
                    prompt=f"Prompt: {row['question']}\nOutput A: {row[model_a]}\nOutput B: {row[model_b]}",
                )
                output_a = get_llm_output(
                    scoring_prompt, model=judge, system_prompt=judge_systems_prompt
                )
                # score = self.parse_output(output_a)
                model_outputs.append(output_a)
            judge_outputs.append(model_outputs)
        return judge_outputs


class PreferenceRanker(RelativeRanker):
    """
    Scores by saying which model fits the description better
    """

    def extract_scores(self, output):
        """parse out the score from the output of the following format
        Analysis: {{reasoning}}
        Model: {{A or B}}
        """
        output = output.replace("Output ", "").replace("output ", "")
        output = re.sub(r"[#*]", "", output)
        # ignore spaces
        score_pattern = re.compile(r"Model: (A|B|tie)", re.IGNORECASE | re.MULTILINE)
        score = score_pattern.findall(output)
        # apply end_of_output parse if necessary
        end_of_output = output[-20:]
        end_of_out_pattern = re.compile(r"\b(A|B|tie)\b", re.IGNORECASE | re.MULTILINE)
        try:
            if len(score) == 0:
                score = end_of_out_pattern.findall(end_of_output)
            if score[0] == "A" or score[0] == "a":
                return 1
            elif score[0] == "B" or score[0] == "b":
                return -1
            elif score[0] == "tie" or score[0] == "Tie":
                return 0
            else:
                print(f"Invalid score: {score[0]}")
                return 0
        except:
            print(f"Invalid score: {score}")
            return 0

    def get_score(self, row, axis, dummy_eval=False):
        if dummy_eval:
            rand = np.random.rand()
            if rand < 0.33:
                return ["Analysis: Because I said so\nModel: A"] * len(self.args.models)
            elif rand < 0.66:
                return ["Analysis: Because I said so\nModel: B"] * len(self.args.models)
            else:
                return ["Analysis: Because I said so\nModel: tie"] * len(
                    self.args.models
                )

        prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants (A and B) to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 

        Here is the prompt and the outputs of A and B respectively:

        {prompt}

        Please respond with the model which contains a higher quality response. Based on your analysis, please explain your reasoning before assigning a score. Use the following format for your response:
        Analysis: {{reasoning}}
        Model: {{A, B, tie}}
        """

        judge_systems_prompt = "You are a fair and objective judge of model outputs. Your evaluations are clear, concise, and free from exaggerative language. You strictly adhere to the format and guidelines provided by the user, ensuring each decision is well-supported by the evidence within the outputs themselves."
        judge_outputs = []
        for judge in self.args.judges:
            # print(f"Getting judgement for Judge = {judge}")
            model_outputs = []
            for model in self.args.models:
                model_a, model_b = model, [m for m in self.args.models if m != model][0]
                scoring_prompt = prompt.format(
                    prompt=f"Prompt: {row['question']}\nOutput A: {row[model_a]}\nOutput B: {row[model_b]}"
                )
                output_a = get_llm_output(
                    scoring_prompt, model=judge, system_prompt=judge_systems_prompt
                )
                # score = self.parse_output(output_a)
                model_outputs.append(output_a)
            judge_outputs.append(model_outputs)
        return judge_outputs
