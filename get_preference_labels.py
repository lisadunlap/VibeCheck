from utils import parse_bullets
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.cache import CacheConfig, CacheType, CacheFactory
import numpy as np
import pandas as pd
import re

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
    parser.add_argument("--data_path", type=str, default="data/helm/claude_gemini_gpt_math_cot_bigger.csv")
    parser.add_argument(
        "--models", nargs="+", default=["anthropic/claude-3-5-sonnet-20240620", "openai/gpt-4o-2024-08-06"]
    )
    parser.add_argument("--output_path", type=str, default="data/helm/claude_gemini_gpt_math_cot_bigger_w_pref.csv")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="model to use for judging preference")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)
    lm = LM(model=args.judge_model, cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=True)

    df = pd.read_csv(args.data_path)
    if args.test:
        df = df.head(10)
    df["judge_input"] = df.apply(
        lambda row: f"Prompt: {row['question']}\nOutput A: {row[args.models[0]]}\nOutput B: {row[args.models[1]]}",
        axis=1,
    )
    df["judge_input_reversed"] = df.apply(
        lambda row: f"Prompt: {row['question']}\nOutput A: {row[args.models[1]]}\nOutput B: {row[args.models[0]]}",
        axis=1,
    )
    df = df.sem_map(judge_prompt, return_raw_outputs=True, suffix="preference")
    df = df.sem_map(
        judge_prompt_reversed, return_raw_outputs=True, suffix="preference_reversed"
    )
    df["preference"] = df.apply(lambda row: extract_scores(row["preference"]), axis=1)
    df["preference_reversed"] = df.apply(
        lambda row: extract_scores(row["preference_reversed"]), axis=1
    )
    df["position_bias"] = df["preference_reversed"] == df["preference"]
    df["preference_feature"] = df.apply(
        lambda row: row["preference"] if not row["position_bias"] else 0, axis=1
    )
    df["preference"] = df["preference_feature"].apply(
        lambda x: {"-1": args.models[1], "1": args.models[0], "0": "equal"}[str(x)]
    )
    df["preference_model"] = args.judge_model
    print("Preference counts: ", df.preference.value_counts().to_dict())
    df.to_csv(args.output_path, index=False)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    __main__()
