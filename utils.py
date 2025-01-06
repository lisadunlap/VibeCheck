import os
import pandas as pd
from fuzzywuzzy import fuzz
import wandb

def get_save_str(args, num_samples, model_group):
    # Create string of datapath for saving
    save_str = args.data_path.split("/")[-1].split(".")[0]
    save_str = f"{save_str}/{args.output_name}" if args.output_name else save_str
    save_str = f"{save_str}/{args.proposer}-{args.sampler}_{args.ranker}/{'_'.join(args.judges)}"
    tag = (
        f"{model_group}_k{args.k}_seed{args.seed}"
        if not args.num_samples
        else f"{model_group}_{args.k}_samples{num_samples}_seed{args.seed}"
    )
    tag = f"{tag}_dummy_eval" if args.dummy_eval else tag
    tag = f"{tag}_axes_provided" if args.axes else tag
    tag = f"{tag}_early_stopping" if args.early_stopping else tag
    tag = f"{tag}_filter" if args.filter else tag
    tag = f"{tag}_filter_mm_only" if args.filter_mm_only else tag
    if not os.path.exists(f"{args.save_dir}/{save_str}"):
        os.makedirs(f"{args.save_dir}/{save_str}", exist_ok=True)
    return save_str, tag


def load_experiment(results_dir, tag, args):
    results = pd.read_csv(f"{results_dir}/{tag}-reducer_results.csv")
    eval_axes = (
        results["axis"]
        .value_counts()[: min(args.num_eval, len(results["axis"].unique()))]
        .index.tolist()
    )
    print(f"\n\n{results['axis'].value_counts()}\n{eval_axes}\n\n")
    return eval_axes


def remove_similar_fuzzy_rows(df, col1, col2, threshold=80):
    from fuzzywuzzy import fuzz

    # Create a boolean mask for rows to keep
    mask = [fuzz.ratio(str(a), str(b)) < threshold for a, b in zip(df[col1], df[col2])]
    # Apply the mask to keep only dissimilar rows
    df_filtered = df[mask]
    return df_filtered
