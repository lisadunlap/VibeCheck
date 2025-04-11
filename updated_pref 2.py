from utils import parse_bullets
from utils_llm import get_llm_output
from utils_general import get_from_cache, save_to_cache
import numpy as np
import pandas as pd
import re
import argparse
import openai
import os
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "YOUR OPENAI KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")


judge_prompt = """You are an impartial judge and evaluate the quality of the responses provided by two AI assistants (A and B) to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 

Here is the prompt and the outputs of A and B respectively:

{judge_input}

Please respond with the model which contains a higher quality response. Based on your analysis, please explain your reasoning before assigning a score. Use the following format for your response:
Analysis: {{reasoning}}
Model: {{A, B, tie}}
"""

judge_prompt_reversed = """You are an impartial judge and evaluate the quality of the responses provided by two AI assistants (A and B) to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 

Here is the prompt and the outputs of A and B respectively:

{judge_input_reversed}

Please respond with the model which contains a higher quality response. Based on your analysis, please explain your reasoning before assigning a score. Use the following format for your response:
Analysis: {{reasoning}}
Model: {{A, B, tie}}
"""


def extract_scores(output):
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


import argparse


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/nazcol/VibeCheck/data/llama3-70b-arena/llama_vs_not_llama_with_categories.csv")
    parser.add_argument("--models", nargs="+", default=["human_answers", "chatgpt_answers"])
    parser.add_argument("--output_path", type=str, default="/home/nazcol/VibeCheck/data/llama3-70b-arena/llama_vs_not_llama_with_categories_pref.csv")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model to use for judging preference")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()


    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Data loaded: {len(df)} rows.")

    if args.test:
        df = df.head(10)
        print("Running in test mode: Using first 10 rows.")

    print("Generating judge inputs...")

    judge_inputs = []
    judge_inputs_reversed = []

    for _, row in df.iterrows():
        judge_inputs.append(f"Prompt: {row['question']}\nOutput A: {row[args.models[0]]}\nOutput B: {row[args.models[1]]}")
        judge_inputs_reversed.append(f"Prompt: {row['question']}\nOutput A: {row[args.models[1]]}\nOutput B: {row[args.models[0]]}")

    df["judge_input"] = judge_inputs
    df["judge_input_reversed"] = judge_inputs_reversed

    print("Getting preferences from LLM using 32 threads...")
    preferences = get_llm_output(judge_inputs, args.judge_model)
    preferences_reversed = get_llm_output(judge_inputs_reversed, args.judge_model)

    df["preference"] = preferences
    df["preference_reversed"] = preferences_reversed

    print("Extracting scores...")
    preference_list = df["preference"].tolist()
    preference_reversed_list = df["preference_reversed"].tolist()

    extracted_scores = [extract_scores(x) for x in tqdm(preference_list, desc="Extracting Scores")]
    extracted_scores_reversed = [extract_scores(x) for x in tqdm(preference_reversed_list, desc="Extracting Reversed Scores")]

    df["preference"] = extracted_scores
    df["preference_reversed"] = extracted_scores_reversed

    df["position_bias"] = df["preference_reversed"] == df["preference"]
    df["preference_feature"] = df.apply(lambda row: row["preference"] if not row["position_bias"] else 0, axis=1)
    df["preference"] = df["preference_feature"].apply(lambda x: {"-1": args.models[1], "1": args.models[0], "0": "equal"}[str(x)])
    df["preference_model"] = args.judge_model
    print("Final preference counts:", df["preference"].value_counts().to_dict())

    print("Preference counts:", df.preference.value_counts().to_dict())
    df.to_csv(args.output_path, index=False)
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    __main__()