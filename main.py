import pandas as pd
import os
import random
import numpy as np
import wandb
from omegaconf import OmegaConf
import argparse
from sklearn.metrics import cohen_kappa_score
from itertools import combinations

from serve.utils_llm import get_llm_output
from components.proposer_prompts import *
from components.parsing_utils import *
import components.ranker as rankers
import components.proposer as proposers
import components.reducer as reducers
import components.sampler as samplers
from components.mm_and_pp_modeling import (
    get_score,
    create_feature_df,
    train_model,
    generate_vibe_overlap_heatmap,
    calculate_vibe_overlap
)
from utils import get_save_str, load_experiment, remove_similar_fuzzy_rows


def propose_axes(proposal_df, args):
    proposer = getattr(proposers, args.proposer)(args)
    (
        all_axis_descriptions,
        llm_logs,
        pairwise_differences,
        results,
    ) = proposer.propose(proposal_df)

    # Log the proposer outputs
    wandb.log(
        {
            "per_sample_differences": wandb.Table(dataframe=results),
            "pairwise_diff_llm_logs": wandb.Table(dataframe=llm_logs),
        }
    )
    all_axis_descriptions = list(results["axis_description"])
    return all_axis_descriptions, results


def propose_axes_iteration(proposal_df, args, axes, iteration=1):
    if args.give_prev_axes:
        proposer = getattr(proposers, "LLMProposerIterationNewOnly")(args, axes)
        (
            all_axis_descriptions,
            llm_logs,
            pairwise_differences,
            results,
        ) = proposer.propose(proposal_df)
    else:
        proposer = getattr(proposers, args.proposer)(args)
        (
            all_axis_descriptions,
            llm_logs,
            pairwise_differences,
            results,
        ) = proposer.propose(proposal_df)

    # Log the proposer outputs
    wandb.log(
        {
            f"per_sample_differences-{iteration}": wandb.Table(dataframe=results),
            f"pairwise_diff_llm_logs-{iteration}": wandb.Table(dataframe=llm_logs),
        }
    )
    all_axis_descriptions = list(results["axis_description"])
    return all_axis_descriptions, results


def reduce_axes(all_axis_descriptions, results, args, save_str="results", eval_axes=[]):
    all_axis_descriptions = [x.replace("*", "") for x in all_axis_descriptions]

    reducer = getattr(reducers, args.reducer)(args)
    parent_axes, child_parent_map, tables = reducer.reduce(all_axis_descriptions)
    results["axis"] = child_parent_map

    # Log the reducer outputs
    metrics = {k: wandb.Table(dataframe=v) for k, v in tables.items()}
    metrics["eval_axes"] = results["axis"].value_counts().reset_index()
    wandb.log(metrics)

    prompt = """Here is a list of axes on which two strings may vary. Each axis has a description of what makes a string high or low on that axis.
{existing_axes} 
{new_axes}

It is likely that several of these axes measure similar things. Your task is to remove any redundant axes. Think about if a user would gain any new information from seeing both axes. For example, "Emotional Tone: High: Contains emotionally charged language. Low: Maintains a neutral tone." and "Empathy: High: Shows empathy. Low: Only factual answers without empathy." are redundant because they both measure the emotional content of the text. If two similar axes are found, keep the one that is more informative.

Output the reduced list of axes, seperated by a newline. All of the axes should maintain the same format they have in the list of {{axis}}: High: {{high}} Low: {{low}}

Your Response:"""
    new_axes = list(set(results["axis"]))
    if len(eval_axes) > 0:
        output = get_llm_output(
            prompt.format(
                existing_axes="\n".join(eval_axes), new_axes="\n".join(new_axes)
            ),
            model="gpt-4o",
        )
        print(
            prompt.format(
                existing_axes="\n".join(eval_axes), new_axes="\n".join(new_axes)
            )
        )
        axes = output.replace("*", "").split("\n")
        all_axes = eval_axes + axes
        # remove any axes which do not exist in all_axes
        axes = [a for a in axes if a in all_axes]
        print("------------------------------")
        print(
            f"Old Axes Length: {len(new_axes) + len(eval_axes)}\tNew Axes Length: {len(axes)}"
        )
        print(f"Axes Removed: {set(all_axes) - set(axes)}")
        print("------------------------------")
        # reset new axes to the intersection of the new axes and the axes returned by the user
        new_axes = [n for n in new_axes if n in axes]

    num_eval = min(args.num_eval, len(results["axis"].unique())) if args.num_eval else len(results["axis"].unique())
    eval_axes = (
        results["axis"]
        .value_counts()
        .head(num_eval)
        .index.tolist()
    )

    return eval_axes, results


def evaluate_axes(eval_axes, data_df, args):
    evaluator = getattr(rankers, args.ranker)(args)
    print("Evaluating axes: " + "\n".join(eval_axes))
    metrics, results, scoring_logs = evaluator.score(
        eval_axes, data_df.to_dict("records")
    )

    cohns_results = {}
    for axis in results.axis.unique():
        if "Judge_1_final_score" in results.columns:
            axis_eval = results[results.axis == axis]
            judge_0_score = axis_eval["Judge_0_final_score"]
            judge_1_score = axis_eval["Judge_1_final_score"]
            # compute cohns kappa
            kappa = cohen_kappa_score(judge_0_score, judge_1_score)
            print(f"Kappa for {axis}: {kappa}")
        else:
            kappa = 0

        cohns_results[axis] = kappa

    # results["score"] = results["avg_final_scores"]
    if len(eval_axes) > len(results["axis"].unique()):
        print("Some axes were removed during evaluation.")
    eval_axes = results["axis"].unique().tolist()

    # Calculate vibe overlaps (uncomment if you want to remove any vibes which scores operlap a lot)
    removed_vibes = []
    vibe_overlaps = {}
    for vibe1, vibe2 in combinations(eval_axes, 2):
        overlap = calculate_vibe_overlap(results, vibe1, vibe2, args.models)
        vibe_overlaps[f"{vibe1} & {vibe2}"] = overlap
        if overlap > 0.8:
            print(f"Overlap between {vibe1} and {vibe2}: {overlap}")
            if vibe1 not in removed_vibes and vibe2 not in removed_vibes:
                removed_vibes.append(vibe2)
    eval_axes = [axis for axis in eval_axes if axis not in removed_vibes]

    # Generate and save vibe overlap heatmap
    correlation_matrix, image_path = generate_vibe_overlap_heatmap(
        results, eval_axes, args.models
    )

    # Log the evaluator outputs
    wandb.log(
        {
            "scoring_logs": wandb.Table(dataframe=scoring_logs),
            "summary_results": wandb.Table(dataframe=results),
            "metrics": metrics,
            "vibe_overlap_heatmap": wandb.Image(image_path),
        }
    )
    return eval_axes, metrics, results, scoring_logs, cohns_results

def filter_axes(eval_axes, previous_axes, results, args, iteration=1):
    results["preference"] = results["preference"].apply(
        lambda x: get_pref_score(x, args)
    )
    results["preference"] = results["preference"].apply(get_score)
    # add preference as an axis to results
    preference_df = results.drop_duplicates(subset=["question", *list(args.models)])
    preference_df["axis"] = "preference"
    preference_df["score"] = preference_df["preference"]
    results = pd.concat([results, preference_df])

    feature_df = create_feature_df(results, args).dropna()
    feature_df1 = feature_df.copy()
    feature_df2 = feature_df.copy()
    feature_df1["model_label"] = 0
    feature_df2["model_label"] = 1
    for axis in eval_axes:
        feature_df2[axis] = -feature_df2[axis]
    feature_df2["preference"] = -feature_df2["preference"]
    feature_df = pd.concat([feature_df1, feature_df2])
    feature_df["preference"] = feature_df["preference"].apply(
        lambda x: 0 if x > 0 else 1
    )

    X = feature_df[eval_axes]
    y = feature_df["model_label"]

    # Train and evaluate preference model
    (
        mm_acc,
        mm_feature_importance,
        mm_test_results,
        train_mm_accuracy,
        mm_train_results,
        mm_summary,
        mm_feature_order,
    ) = train_model(
        X, y, X, y, eval_axes, "model_label", model_type="lasso", x_val=True
    )

    # Train and evaluate preference model
    (
        pref_accuracy,
        pref_feature_importance,
        pref_test_results,
        train_pref_accuracy,
        pref_train_results,
        pref_summary,
        pref_feature_order,
    ) = train_model(X, y, X, y, eval_axes, "preference", model_type="lasso", x_val=True)

    # remove 'const' from pref_feature_importance and mm_feature_importance
    pref_feature_importance = pref_feature_importance[
        pref_feature_importance["feature"] != "const"
    ]
    mm_feature_importance = mm_feature_importance[
        mm_feature_importance["feature"] != "const"
    ]
    # abso val of coeff < 0.1
    pref_feature_importance["score"] = pref_feature_importance["feature"].apply(
        lambda x: np.mean(feature_df1[x].tolist())
    )
    pref_feature_importance["pref_score"] = pref_feature_importance["feature"].apply(
        lambda x: np.average(
            [
                s * p
                for s, p in zip(
                    feature_df1[x].tolist(), feature_df1["preference"].tolist()
                )
            ]
        )
    )
    if args.filter_mm_only:
        filtered_eval_axes = pref_feature_importance[
            (pref_feature_importance["score"].abs() > 0.05)
        ]["feature"].tolist()
    else:
        filtered_eval_axes = pref_feature_importance[
            (pref_feature_importance["score"].abs() > 0.05)
            | (pref_feature_importance["pref_score"].abs() > 0.05)
        ]["feature"].tolist()

    filtered_eval_axes = [f for f in filtered_eval_axes if f != "const"]

    removed_axes = list(set(eval_axes) - set(filtered_eval_axes))
    print(f"Removed Axes: {removed_axes}")
    return filtered_eval_axes, removed_axes

from components.mm_and_pp_modeling import prep_feat_df, train_and_evaluate
def train_lr_models(
    eval_results, preference_results, test_eval_results, test_preference_results, args, eval_axes, iteration=1, save_str="results", tag="train", cohns_kapps_results=None
):
    # Prepare feature dataframes
    train_feat_df, train_pref_features = prep_feat_df(eval_results, preference_results, args, eval_axes)
    test_feat_df, test_pref_features = prep_feat_df(test_eval_results, test_preference_results, args, eval_axes)

    # Train preference and model matching (mm) models
    pref_metrics = train_and_evaluate(
        train_pref_features, test_pref_features, eval_axes, "preference", args, tag, iteration, model_type="lasso" if args.lasso_regression else "logistic"
    )
    mm_metrics = train_and_evaluate(
        train_pref_features, test_pref_features, eval_axes, "model_label", args, tag, iteration, model_type="lasso" if args.lasso_regression else "logistic"
    )

    # Compute feature importance scores
    all_feature_importance = pref_metrics["feature_importance"].merge(
        mm_metrics["feature_importance"], on="feature", suffixes=("_pref", "_mm")
    )
    all_feature_importance["cohns"] = all_feature_importance["feature"].apply(
        lambda x: cohns_kapps_results.get(x, 0)
    )

    # Log results to WandB
    wandb.log({
        f"all_feature_importance_iter_{iteration}-{tag}": wandb.Table(dataframe=all_feature_importance),
    })

    return (
        pref_metrics["accuracy"],
        mm_metrics["accuracy"],
        pref_metrics["test_results"],
        mm_metrics["test_results"],
        list(all_feature_importance["feature"]),
        pref_metrics["feature_importance"],
        mm_metrics["feature_importance"],
        mm_metrics["feature_order"],
    )


def get_misclassified_samples(
    pref_train_results, mm_train_results, data_df, models, args
):
    pref_train_results["pref_correct"] = pref_train_results["correct"]
    mm_train_results["mm_correct"] = mm_train_results["correct"]
    pref_train_results = pref_train_results.drop_duplicates(
        subset=["question"] + models
    )

    pref_train_results = pref_train_results.merge(
        mm_train_results, on=["question"] + models, how="inner"
    )
    if args.filter_mm_only:
        misclassified = pref_train_results[pref_train_results["mm_correct"] == False]
    else:
        misclassified = pref_train_results[
            (pref_train_results["pref_correct"] == False)
            | (pref_train_results["mm_correct"] == False)
        ]

    # get those samples from data_df
    print(f"Misclassified: {len(misclassified)}")
    # return samples where either models are misclassified
    return data_df[data_df["question"].isin(misclassified["question"])].drop_duplicates(
        subset="question"
    )


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

    args_copy = args.copy()
    args_copy.judges = args.preference_judges
    evaluator = getattr(rankers, "PreferenceRanker")(args_copy)

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
    return preferences


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
    df.drop_duplicates(subset=args.models, inplace=True)

    df = (
        df[["question", *args.models]]
        if "preference" not in df.columns
        else df[["question", *args.models, "preference"]]
    )

    if args.test_data_path:
        heldout_df = pd.read_csv(args.test_data_path)
        heldout_df = (
            heldout_df[["question", *args.models]]
            if "preference" not in heldout_df.columns
            else heldout_df[["question", *args.models, "preference"]]
        )
    else:
        heldout_df = df

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
        name=args.output_name if args.output_name else "plz_name_me",
    )
    wandb.run.log_code(flags.config)

    num_samples =  min(args.num_samples, df.shape[0]) if args.num_samples else df.shape[0]
    num_eval_samples = (
        min(args.num_eval_samples, heldout_df.shape[0]) 
        if args.num_eval_samples else heldout_df.shape[0]
    )
    if args.test:
        num_samples, num_eval_samples = 10, 10

    save_str, tag = get_save_str(args, num_samples, model_group)
    early_stopping = args.early_stopping if hasattr(args, "early_stopping") else False

    df = remove_similar_fuzzy_rows(df, args.models[0], args.models[1], threshold=80)
    heldout_df = remove_similar_fuzzy_rows(
        heldout_df, args.models[0], args.models[1], threshold=80
    )

    # Randomly sample rows
    if args.num_samples or args.test:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)[
            :num_samples
        ]
    if args.num_eval_samples or args.test:
        heldout_df = heldout_df.sample(frac=1, random_state=args.seed).reset_index(
            drop=True
        )[:num_eval_samples]

    # Sample (this is a NoOp if you arent clustering by question type)
    sampler = getattr(samplers, args.sampler)(args)
    df, _ = sampler.sample(df)

    # Initialize variables for the iteration loop
    max_iterations = args.max_iterations if hasattr(args, "max_iterations") else 3

    # Initialize preference results if needed
    if "preference" not in df.columns:
        enter = input(
            "You are about to generate preference scores. Press enter to continue."
        )
        if not enter == "":
            exit()

        df["preference"] = get_llm_pref_score(df, args)
        heldout_df["preference"] = get_llm_pref_score(heldout_df, args)
        df = df[df["preference"] != "equal"]
        heldout_df = heldout_df[heldout_df["preference"] != "equal"]
        df.to_csv(f"{args.save_dir}/{save_str}/df-{tag}.csv", index=False)
        heldout_df.to_csv(
            f"{args.save_dir}/{save_str}/heldout_df-{tag}.csv", index=False
        )
        
    print(f"Train Preference Value Counts: {df['preference'].value_counts()}")
    print(
        f"Test Preference Value Counts: {heldout_df['preference'].value_counts()}"
    )

    eval_axes = []
    prev_eval_axes = []
    max_mm_acc = 0
    max_pref_acc = 0
    eval_axes_per_iter_table = pd.DataFrame()
    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===\n")
        if args.eval_only:
            max_iterations = 1
            if args.axes:
                eval_axes = list(args.axes)
                filtered_eval_axes, metrics, eval_results, scoring_logs, cohns_kappa = (
                    evaluate_axes(eval_axes, df, args)
                )
            else:
                args.early_stopping = early_stopping
                eval_axes = load_experiment(args.save_dir, tag, args)
                filtered_eval_axes, metrics, eval_results, scoring_logs, cohns_kappa = (
                    evaluate_axes(eval_axes, df, args)
                )
                eval_results.to_json(
                    f"{args.save_dir}/{save_str}/eval_results_{iteration}.json",
                    orient="records",
                )
        else:
            # Step 1: Propose Axes
            if iteration == 0:
                if not args.axes:
                    proposal_df = df.sample(
                        args.num_proposal_samples
                    )  # Use the entire dataset initially
                    all_axis_descriptions, proposer_results = propose_axes(
                        proposal_df, args
                    )
            else:
                # Use misclassified samples from previous iteration
                proposal_df = get_misclassified_samples(
                    pref_train_results, mm_train_results, df, list(args.models), args
                )
                proposal_df = proposal_df.sample(min(args.num_proposal_samples, len(proposal_df)))
                if proposal_df.empty:
                    print("No misclassified samples. Stopping iterations.")
                    break
                all_axis_descriptions, proposer_results = propose_axes_iteration(
                    proposal_df, args, eval_axes, iteration=iteration
                )

            # Step 2: Reduce Axes (start with eval axes if provided)
            if not (args.axes and iteration == 0):
                reduced_eval_axes, reducer_results = reduce_axes(
                    all_axis_descriptions, proposer_results, args, save_str, eval_axes
                ) 
                reducer_results.to_json(
                    f"{args.save_dir}/{save_str}/reducer_results-{iteration}.json",
                    orient="records",
                )
                print(f"adding {reduced_eval_axes} to {eval_axes}")
                eval_axes = list(set(eval_axes + reduced_eval_axes))
            else:
                prev_eval_axes = eval_axes
                eval_axes = list(args.axes)
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print("Eval Axes: ", "\n".join(eval_axes))
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            if args.proposer_only:
                exit(0)

            # Step 3: Evaluate Axes
            # Evaluate on the entire dataset (df)
            args.early_stopping = early_stopping
            filtered_eval_axes, metrics, eval_results, scoring_logs, cohns_kappa = (
                evaluate_axes(eval_axes, df, args)
            )
            eval_results.to_json(
                f"{args.save_dir}/{save_str}/eval_results_{iteration}.json",
                orient="records",
            )

        if args.filter or args.filter_mm_only:
            # # Step 3.5: Filter Axes
            filtered_eval_axes, removed_axes = filter_axes(
                filtered_eval_axes, prev_eval_axes, eval_results, args
            )
        prev_eval_axes = filtered_eval_axes

        # # Step 4: Train LR Models for both mm and preference
        preference_results = df.copy()
        preference_results["avg_final_scores"] = preference_results["preference"].apply(lambda x: get_score(get_pref_score(x, args)))
        preference_results["axis"] = "preference"
        test_preference_results = heldout_df.copy()
        test_preference_results["avg_final_scores"] = test_preference_results["preference"].apply(lambda x: get_score(get_pref_score(x, args)))
        test_preference_results["axis"] = "preference"

        print(f"Filtered Eval Axes: {filtered_eval_axes}")
        (
            pref_accuracy,
            mm_accuracy,
            pref_train_results,
            mm_train_results,
            filtered_eval_axes,
            pref_feature_importance,
            mm_feature_importance,
            mm_feature_order,
        ) = train_lr_models(
            eval_results,
            preference_results,
            eval_results,
            preference_results,
            args,
            filtered_eval_axes,
            iteration + 1,
            save_str,
            "train",
            cohns_kappa,
        )

        # convert into numbered list for vibes
        eval_axes_str = "\n".join(
            [f"{i+1}. {axis}" for i, axis in enumerate(eval_axes)]
        )
        filtered_eval_axes_str = "\n".join(
            [f"{i+1}. {axis}" for i, axis in enumerate(filtered_eval_axes)]
        )
        eval_axes_per_iter_table = pd.concat(
            [
                eval_axes_per_iter_table,
                pd.DataFrame(
                    {
                        "iteration": iteration + 1,
                        "eval_axes": eval_axes_str,
                        "filtered_eval_axes": filtered_eval_axes_str,
                    },
                    index=[0],
                ),
            ],
            axis=0,
        )
        # log train metrics
        wandb.log(
            {
                "iteration": iteration + 1,
                "pref_accuracy": pref_accuracy,
                "vibes": wandb.Table(dataframe=eval_axes_per_iter_table),
                "mm_accuracy": mm_accuracy,
            }
        )
        if pref_accuracy > max_pref_acc:
            max_pref_acc = pref_accuracy
            max_iteration = iteration
        wandb.run.summary["max_mm_accuracy"] = max_mm_acc
        wandb.run.summary["max_pref_accuracy"] = max_pref_acc
        wandb.run.summary["max_iteration"] = max_iteration + 1
        wandb.run.summary["dataset_len"] = len(df)
        wandb.run.summary["heldout_dataset_len"] = len(heldout_df)
        wandb.run.summary["eval_axes"] = eval_axes

        if args.eval_every_iteration:
            # eval on test set
            args.early_stopping = False
            _, test_metrics, test_eval_results, test_scoring_logs, test_cohns_kappa = (
                evaluate_axes(eval_axes, heldout_df, args)
            )

            (
                test_pref_accuracy,
                test_mm_accuracy,
                test_pref_train_results,
                test_mm_train_results,
                _, _, _, _,
            ) = train_lr_models(
                eval_results,
                preference_results,
                test_eval_results,
                test_preference_results,
                args,
                filtered_eval_axes,
                iteration + 1,
                save_str,
                "test",
                test_cohns_kappa,
            )

            # save eval_results and test_eval_results
            eval_results.to_json(
                f"{args.save_dir}/{save_str}/eval_results_{iteration}.json",
                orient="records",
            )
            test_eval_results.to_json(
                f"{args.save_dir}/{save_str}/test_eval_results_{iteration}.json",
                orient="records",
            )
            preference_results.to_json(
                f"{args.save_dir}/{save_str}/preference_results.json", orient="records"
            )
            test_preference_results.to_json(
                f"{args.save_dir}/{save_str}/test_preference_results.json",
                orient="records",
            )

            # Step 5: Check for Convergence
            if mm_accuracy >= 0.99:
                print(
                    f"Convergence achieved with MM accuracy {pref_accuracy:.2f}. Stopping iterations."
                )
                break

            wandb.log(
                {
                    "iteration": iteration + 1,
                    "test_pref_accuracy": test_pref_accuracy,
                    "test_mm_accuracy": test_mm_accuracy,
                }
            )
            wandb.run.summary["test_mm_accuracy"] = test_mm_accuracy
            wandb.run.summary["test_pref_accuracy"] = test_pref_accuracy
            wandb.run.summary["test_id_mm_accuracy"] = test_easy_mm_accuracy
            wandb.run.summary["test_id_pref_accuracy"] = test_easy_pref_accuracy

    if args.filter_p_values:
        mm_eval_axes = mm_feature_importance[
            (mm_feature_importance["p_value"] < 0.05)
            & (mm_feature_importance["score"].abs() > 0.05)
        ]["feature"].tolist()
        pref_eval_axes = pref_feature_importance[
            (pref_feature_importance["p_value"] < 0.05)
            & (pref_feature_importance["score"].abs() > 0.05)
        ]["feature"].tolist()
        if args.filter_mm_only:
            final_eval_axes = mm_eval_axes
        elif args.filter_pref_only:
            final_eval_axes = pref_eval_axes
        else:
            # get intersection of both
            final_eval_axes = list(set(mm_eval_axes).intersection(set(pref_eval_axes)))
    elif args.num_final_eval:
        # select by LARS order
        final_eval_axes = mm_feature_order[: args.num_final_eval]
    else:
        final_eval_axes = eval_axes

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(f"Final Eval Axes: {final_eval_axes}")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    wandb.summary["final_eval_axes"] = final_eval_axes
    wandb.summary["all_eval_axes"] = eval_axes

    (
        pref_accuracy,
        mm_accuracy,
        pref_train_results,
        mm_train_results,
        filtered_eval_axes,
        pref_feature_importance,
        mm_feature_importance,
        mm_feature_order,
    ) = train_lr_models(eval_results,
        preference_results,
        eval_results,
        preference_results,
        args,
        final_eval_axes,
        "final",
        save_str,
        "train",
        cohns_kappa,
    )

    # convert into numbered list for vibes
    eval_axes_str = "\n".join([f"{i+1}. {axis}" for i, axis in enumerate(eval_axes)])
    filtered_eval_axes_str = "\n".join(
        [f"{i+1}. {axis}" for i, axis in enumerate(final_eval_axes)]
    )
    eval_axes_per_iter_table = pd.concat(
        [
            eval_axes_per_iter_table,
            pd.DataFrame(
                {
                    "iteration": iteration + 1,
                    "eval_axes": eval_axes_str,
                    "filtered_eval_axes": filtered_eval_axes_str,
                },
                index=[0],
            ),
        ],
        axis=0,
    )
    # log train metrics
    wandb.log(
        {
            "iteration": iteration + 1,
            "pref_accuracy": pref_accuracy,
            "vibes": wandb.Table(dataframe=eval_axes_per_iter_table),
            "mm_accuracy": mm_accuracy,
        }
    )

    # eval on test set
    args.early_stopping = False
    _, test_metrics, test_eval_results, test_scoring_logs, test_cohns_kappa = (
        evaluate_axes(eval_axes, heldout_df, args)
    )

    (
        test_pref_accuracy,
        test_mm_accuracy,
        test_pref_train_results,
        test_mm_train_results,
        _, _, _, _,
    ) = train_lr_models(
        eval_results,
        preference_results,
        test_eval_results,
        test_preference_results,
        args,
        final_eval_axes,
        "final",
        save_str,
        "test",
        test_cohns_kappa,
    )

    # save eval_results and test_eval_results
    eval_results.to_json(
        f"{args.save_dir}/{save_str}/eval_results_final.json", orient="records"
    )
    test_eval_results.to_json(
        f"{args.save_dir}/{save_str}/test_eval_results_final.json", orient="records"
    )
    preference_results.to_json(
        f"{args.save_dir}/{save_str}/preference_results.json", orient="records"
    )
    test_preference_results.to_json(
        f"{args.save_dir}/{save_str}/test_preference_results.json", orient="records"
    )

    wandb.summary["final_test_pref_accuracy"] = test_pref_accuracy
    wandb.summary["final_test_mm_accuracy"] = test_mm_accuracy
    print(f"Final Test Pref Accuracy: {test_pref_accuracy}")
    print(f"Final Test MM Accuracy: {test_mm_accuracy}")


if __name__ == "__main__":
    main()
