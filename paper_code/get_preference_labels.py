import pandas as pd
import os
import json
import random
import numpy as np
import wandb
from omegaconf import OmegaConf
import argparse

from components.proposer_prompts import *
from components.parsing_utils import *
import components.ranker as rankers
from components.mm_and_pp_modeling import get_score
from utils import get_save_str


def get_pref_score(preference, args):
    if preference == args.models[0]:
        return [1, -1]
    elif preference == args.models[1]:
        return [-1, 1]
    else:
        return [0, 0]


def get_llm_pref_score(df, args):
    if args.dummy_preference:
        return [args.models[0]] * len(df)

    args.judges = args.preference_judges
    evaluator = getattr(rankers, "PreferenceRanker")(args)

    # Score preference on training data
    (
        preference_metrics,
        preference_results,
        preference_scoring_logs,
    ) = evaluator.score(
        ["preference"],
        df.to_dict("records"),
    )
    preference_results["score"] = preference_results["avg_final_scores"]

    def get_p(score):
        if score > 0:
            return args.models[0]
        elif score < 0:
            return args.models[1]
        else:
            return "equal"

    preferences = preference_results["score"].apply(get_p)
    print(f"Preferences: {preferences}")
    return preferences, preference_results


def main():
    # Add in args to override defaults
    parser = argparse.ArgumentParser(description="CLIP Advice")
    parser.add_argument("--config", default="configs/base.yaml", help="config file")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values "
        "(use dots for.nested=overrides)",
    )
    flags, unknown = parser.parse_known_args()

    overrides = OmegaConf.from_cli(flags.overrides)
    base_cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.load(flags.config)
    args = OmegaConf.merge(base_cfg, cfg, overrides)
    args.yaml = flags.config

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Turn off wandb logging if not needed
    if not args.wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    proj_name = args.project if not args.dummy_eval else f"llm_eval_refactor_debug"
    proj_name = f"{proj_name}_test" if args.test else proj_name
    df = pd.read_csv(args.data_path)
    print(f"Models: {args.models}")
    print(f"Eval Axes: {args.axes}")
    print(f"df columns: {df.columns}")
    # Remove duplicate question-answer
    df.drop_duplicates(subset=args.models, inplace=True)

    if args.group_column:
        groups = df[args.group_column].unique()
        print(f"Running VibeCheck on group {args.group_column}({groups})")
        print(f"Group value counts: {df[args.group_column].value_counts()}")
    else:
        groups = ["all"]

    old_df = df
    print(f"df columns: {df.columns}")
    df = (
        df[["question", *args.models]]
        if "preference" not in df.columns
        else df[["question", *args.models, "preference"]]
    )
    # Add in group_column if it exists
    if args.group_column:
        df[args.group_column] = old_df[args.group_column]

    if args.test_data_path:
        heldout_df = pd.read_csv(args.test_data_path)
        heldout_df = (
            heldout_df[["question", *args.models]]
            if "preference" not in heldout_df.columns
            else heldout_df[["question", *args.models, "preference"]]
        )
    else:
        heldout_df = df

    # Get first 3 letters of each model if length is too long (>50)
    model_group = "-".join(args.models).replace(" ", "")
    model_group = (
        "-".join([x[:3] for x in args.models]).replace(" ", "")
        if len(model_group) > 50
        else model_group
    )
    wandb.init(
        project=proj_name,
        config=dict(args),
        group=model_group,
        name=args.output_name if args.output_name else "preference_labels",
    )
    wandb.run.log_code(flags.config)

    num_samples = (
        min(args.num_samples, df.shape[0]) if args.num_samples else df.shape[0]
    )
    num_samples = 10 if args.test else num_samples
    num_eval_samples = (
        min(args.num_eval_samples, heldout_df.shape[0])
        if args.num_eval_samples
        else heldout_df.shape[0]
    )
    num_eval_samples = 10 if args.test else num_eval_samples

    save_str, tag = get_save_str(args, num_samples, model_group)

    # Randomly sample rows
    if args.num_samples or args.test:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)[
            :num_samples
        ]
    if args.num_eval_samples or args.test:
        heldout_df = heldout_df.sample(frac=1, random_state=args.seed).reset_index(
            drop=True
        )[:num_eval_samples]

    if "preference" not in df.columns:
        df["preference"], pref_results = get_llm_pref_score(df, args)
        heldout_df["preference"], heldout_pref_results = get_llm_pref_score(
            heldout_df, args
        )
        # drop any preference
        print(f"Saving to {args.data_path.replace('.csv', '_with_pref.csv')}")
        df.to_csv(f"{args.data_path.replace('.csv', '_with_pref.csv')}", index=False)
        # print(f"Saving to {args.test_data_path.replace('.csv', '_with_pref.csv')}")
        # heldout_df.to_csv(
        #     f"{args.test_data_path.replace('.csv', '_with_pref.csv')}", index=False
        # )

        print(f"Value Counts: {df['preference'].value_counts()}")
        print(f"Value Counts: {heldout_df['preference'].value_counts()}")
        for model in args.models:
            wandb.summary[f"{model}_pref_count"] = df["preference"].value_counts(
                normalize=True
            )[model]
            wandb.summary[f"{model}_heldout_pref_count"] = heldout_df[
                "preference"
            ].value_counts(normalize=True)[model]

    def get_winrate(scores, exclude_ties=False):
        scores = np.array(scores)
        scores = np.array([1 if x > 0 else 0 if x < 0 else 0.5 for x in scores])
        if exclude_ties:
            return np.sum(scores) / (np.sum(scores != 0.5))
        # 1 if model 1 wins, -1 if model 2 wins, 0.5 if tie
        return np.mean(scores)

    # Print preference distribution in a clear format
    print("\n=== Model Preference Distribution ===")
    print("Training Data:")
    train_counts = df["preference"].value_counts()
    train_total = len(df)
    for model, count in train_counts.items():
        percentage = (count / train_total) * 100
        print(f"{model}: {count:,} samples ({percentage:.1f}%)")

    print("\nTest Data:")
    test_counts = heldout_df["preference"].value_counts()
    test_total = len(heldout_df)
    for model, count in test_counts.items():
        percentage = (count / test_total) * 100
        print(f"{model}: {count:,} samples ({percentage:.1f}%)")
    print("===================================\n")

    wandb.summary["num_samples"] = num_samples
    wandb.summary["num_eval_samples"] = num_eval_samples
    wandb.summary[f"{args.models[0]} winrate"] = get_winrate(
        pref_results["avg_final_scores"].tolist()
    )
    wandb.summary[f"{args.models[0]} heldout_winrate"] = get_winrate(
        heldout_pref_results["avg_final_scores"].tolist()
    )
    wandb.summary[f"{args.models[0]} winrate_exclude_ties"] = get_winrate(
        pref_results["avg_final_scores"].tolist(), exclude_ties=True
    )
    wandb.summary[f"{args.models[0]} heldout_winrate_exclude_ties"] = get_winrate(
        heldout_pref_results["avg_final_scores"].tolist(), exclude_ties=True
    )


if __name__ == "__main__":
    main()
