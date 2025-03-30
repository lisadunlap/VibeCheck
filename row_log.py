from utils import parse_bullets
from recent_utils_llm  import get_llm_output
from utils_general import get_from_cache, save_to_cache
import numpy as np
import pandas as pd
import re
import argparse
import openai
import os
import wandb
import time
from pathlib import Path
import json
from tqdm import tqdm
import vllm
from vllm import LLM, SamplingParams
import logging
from datetime import datetime


#config for VLLM 
sampling_params = SamplingParams(
    temperature=0.1,#more deterministic
    top_p=0.95,#more deterministic - choses likely tokens
    max_tokens=512#max tokens
)

def setup_llm(model_path):
    """Initialize the vLLM model."""
    print(f"Loading model from {model_path}...")
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

    return LLM(model=model_path, dtype="float16", gpu_memory_utilization=0.9, max_model_len=2048 ,trust_remote_code=True)



judge_prompt = """You are an impartial judge and evaluate the quality of the responses provided by two AI assistants (A and B) to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 

Here is the prompt and the outputs of A and B respectively:

{judge_input}

Please respond with the model which contains a higher quality response. Based on your analysis, please explain your reasoning before assigning a score. Use the following format for your response:
Analysis: {{reasoning}}
Model: {{A, B, tie}}
"""

judge_prompt_reversed = """You are an impartial judge and evaluate the quality of the responses provided by two AI assistants (A and B) to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 

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

class LlamaJudge:
    def __init__(self, server_url="http://localhost:8001"):
        self.server_url = f"{server_url}/v1/completions"
        self.headers = {"Content-Type": "application/json"}

    def judge_responses(self, response_a, response_b, criteria):
        prompt = f"""You are a fair and unbiased judge. Compare these two responses based on {criteria}:

Response A: {response_a}
Response B: {response_b}

Which response better satisfies the criteria of {criteria}? Explain your reasoning.

Your judgment:"""

        data = {
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.1  # Lower temperature for more consistent judgments
        }

        try:
            response = requests.post(
                self.server_url,
                headers=self.headers,
                json=data
            )
            result = response.json()
            return result['choices'][0]['text']
        except Exception as e:
            logger.error(f"Error in judging: {e}")
            return None

# Add after imports
logging.basicConfig(
    filename=f'llama_judge_inputs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/nazcol/VibeCheck/data/llama3-70b-arena/llama_vs_not_llama_creative_writing.csv")
    parser.add_argument("--models", nargs="+", default=["llama-3-70b-instruct", "not_llama"])
    parser.add_argument("--output_path", type=str, default="/home/nazcol/VibeCheck/data/llama3-70b-arena/llama_vs_not_llama_cw_llama_judge.csv")
    parser.add_argument("--judge_model", type=str, default="llama-8b", help="Model to use for judging preference")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    #cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
    #cache = CacheFactory.create_cache(cache_config)
    #lm = LM(model="gpt-4o", cache=cache)
    #lotus.settings.configure(lm=lm, enable_cache=True)

    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Data loaded: {len(df)} rows.")

    if args.test:
        df = df.head(10)
        print("Running in test mode: Using first 10 rows.")

    try:
        # Initialize wandb first
        run = wandb.init(
            project="naz_vibecheck_trigger",
            entity="clipinvariance",
            name=f"gpt creative writing {args.models[0]}_vs_{args.models[1]}",
            config={
                "data_path": args.data_path,
                "models": args.models,
                "judge_model": args.judge_model,
                "test_mode": args.test
            }
        )

        print("Generating judge inputs...")
        judge_inputs = []
        judge_inputs_reversed = []

        # Log inputs
        logging.info("\n=== SAMPLE INPUTS ===\n")
        for idx, row in df.iterrows():
            normal_input = f"Prompt: {row['question']}\nOutput A: {row[args.models[0]]}\nOutput B: {row[args.models[1]]}"
            reversed_input = f"Prompt: {row['question']}\nOutput A: {row[args.models[1]]}\nOutput B: {row[args.models[0]]}"
            
            judge_inputs.append(normal_input)
            judge_inputs_reversed.append(reversed_input)
            
            if idx < 5:
                logging.info(f"\n--- Example {idx+1} ---")
                logging.info(f"\nNormal Order Input:\n{normal_input}")
                logging.info(f"\nReversed Order Input:\n{reversed_input}")

        print("Getting preferences from LLM...")
        preferences = get_llm_output(judge_inputs, args.judge_model)
        preferences_reversed = get_llm_output(judge_inputs_reversed, args.judge_model)

        # Store in DataFrame
        df["judge_input"] = judge_inputs
        df["judge_input_reversed"] = judge_inputs_reversed
        df["preference"] = preferences
        df["preference_reversed"] = preferences_reversed

        print("Extracting scores...")
        extracted_scores = [extract_scores(x) for x in tqdm(df["preference"].tolist(), desc="Extracting Scores")]
        extracted_scores_reversed = [extract_scores(x) for x in tqdm(df["preference_reversed"].tolist(), desc="Extracting Reversed Scores")]

        df["preference_score"] = extracted_scores
        df["preference_reversed_score"] = extracted_scores_reversed
        
        # Calculate position bias
        df["position_bias"] = df["preference_score"] == df["preference_reversed_score"]
        position_bias_count = df["position_bias"].sum()
        total_samples = len(df)
        bias_percentage = (position_bias_count / total_samples) * 100

        # Calculate final preferences
        df["preference_feature"] = df.apply(
            lambda row: row["preference_score"] if not row["position_bias"] else 0, 
            axis=1
        )
        
        df["preference"] = df["preference_feature"].apply(
            lambda x: {"-1": args.models[1], "1": args.models[0], "0": "equal"}[str(x)]
        )

        # Save to CSV before wandb logging
        print("Saving results to CSV...")
        df.to_csv(args.output_path, index=False)
        print(f"Saved to {args.output_path}")

        # Log to wandb
        print("Logging to wandb...")
        preference_counts = df["preference"].value_counts().to_dict()
        
        wandb.log({
            "position_bias_percentage": bias_percentage,
            "position_bias_count": position_bias_count,
            "total_samples": total_samples,
            "position_bias_rate": df["position_bias"].mean(),
            "preference_counts": preference_counts
        })

        # Log win rates
        for model in args.models + ["equal"]:
            if model in preference_counts:
                wandb.log({
                    f"{model}_win_rate": preference_counts[model] / len(df)
                })

        # Create visualization
        print("Creating wandb visualizations...")
        
        # Create a more detailed preference distribution table
        preference_counts = df["preference"].value_counts().to_dict()
        
        # Ensure all models (including equal) are in the counts
        all_models = args.models + ["equal"]
        complete_counts = {model: preference_counts.get(model, 0) for model in all_models}
        
        # Create table data with all models
        table_data = [
            [model, count, (count/len(df))*100] 
            for model, count in complete_counts.items()
        ]
        
        # Create wandb table with percentages
        table = wandb.Table(
            data=table_data,
            columns=["model", "count", "percentage"]
        )
        
        # Log both raw counts and visualization
        wandb.log({
            "preference_distribution_table": table,
            "preference_counts_raw": complete_counts,
            "preference_distribution": wandb.plot.bar(
                table,
                "model",
                "count",
                title="Model Preference Distribution"
            )
        })

        # Print counts to terminal for immediate feedback
        print("\nPreference Distribution:")
        for model, count in complete_counts.items():
            percentage = (count/len(df))*100
            print(f"{model}: {count} ({percentage:.2f}%)")

        # Log examples
        detailed_examples = []
        for i in range(min(10, len(df))):
            detailed_examples.append({
                "question": df.iloc[i]['question'],
                f"output_{args.models[0]}": df.iloc[i][args.models[0]],
                f"output_{args.models[1]}": df.iloc[i][args.models[1]],
                "judge_input": df.iloc[i]["judge_input"],
                "judge_response": df.iloc[i]["preference"],
                "judge_input_reversed": df.iloc[i]["judge_input_reversed"],
                "judge_response_reversed": df.iloc[i]["preference_reversed"],
                "preference": df.iloc[i]["preference"],
                "position_bias": df.iloc[i]["position_bias"],
                "preference_score": df.iloc[i]["preference_score"],
                "preference_reversed_score": df.iloc[i]["preference_reversed_score"]
            })
        wandb.log({"detailed_examples": detailed_examples})

        # Create a detailed table with all rows, not just summary
        print("Creating detailed wandb table...")
        
        detailed_table_data = []
        for idx, row in df.iterrows():
            detailed_table_data.append([
                idx,  # row number
                row['question'],  # the prompt
                row[args.models[0]],  # first model's response
                row[args.models[1]],  # second model's response
                row['preference'],  # which model was preferred
                row['position_bias'],  # whether position bias was detected
                row['preference_score'],  # original score
                row['preference_reversed_score'],  # reversed score
            ])
        
        # Create detailed wandb table
        detailed_table = wandb.Table(
            data=detailed_table_data,
            columns=[
                "index",
                "question",
                f"{args.models[0]}_response",
                f"{args.models[1]}_response",
                "preferred_model",
                "position_bias",
                "original_score",
                "reversed_score"
            ]
        )
        
        # Log both summary and detailed tables
        wandb.log({
            "all_comparisons": detailed_table,  # detailed table with all rows
            "preference_distribution_table": table  # keep existing summary table
        })

        # Finish wandb run
        wandb.finish()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logging.error(f"Error occurred: {str(e)}")
        if wandb.run is not None:
            wandb.finish()
        raise e

if __name__ == "__main__":
    __main__()
