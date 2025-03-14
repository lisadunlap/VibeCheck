import argparse
import wandb
import pandas as pd
from plotly import graph_objects as go
import numpy as np
import os
from omegaconf import OmegaConf

from typing import List

from utils import (
    create_gradio_app,
    train_embedding_classifier,
)
from components.utils_llm import get_llm_embedding, get_llm_output

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from functools import wraps

def timeit(func):
    """Decorator to measure the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper

def get_vibe_question_types(vibe_df: pd.DataFrame, config: OmegaConf, batch_size: int = 50):
    """Describe what types of questions result in high scores for a given vibe."""

    filtered_vibe_df = vibe_df[vibe_df["score"].abs() > 0.0].copy()
    vibin_questions = filtered_vibe_df.sort_values(
        ["vibe", "score"], ascending=[True, False]
    )

    prompt = """You are a machine learning researcher trying to discover what types of questions result in a model exhibiting a certain behavior. Given the following behavior and a list of questions along with a score of how much the model exhibits the behavior on that question, describe what types of questions result in the model exhibiting the behavior. Each score is between 1 and 5, where 1 is does not exhibit the behavior at all, and 5 is exhibits the behavior completely.

{input_text}

Please respond with a 1 sentence description of what types of questions result in the model exhibiting the behavior and what types of questions result in the model not exhibiting the behavior. This should be a single sentence that is human interpretable and concise, such that I can create a diverse set of new questions that result in the model exhibiting the behavior as well as new questions that result in the model not exhibiting the behavior.

Format your response in the following format:
Question types which exhibit the behavior: <description>
Question types which do not exhibit the behavior: <description>
"""
    new_df = []
    for vibe in vibin_questions["vibe"].unique():
        single_vibe_df = vibin_questions[vibin_questions["vibe"] == vibe].copy()
        # Sample and create input text for each vibe
        sampled_df = single_vibe_df.sample(min(batch_size, len(single_vibe_df)))
        input_texts = sampled_df.apply(
            lambda row: f"Behavior: {row['vibe']}\nQuestion: {row['question']}\nScore: {row['score']}",
            axis=1,
        ).tolist()
        input_text = "\n-------------\n".join(input_texts)
        new_df.append({"vibe": vibe, "input_text": input_text})

    new_df = pd.DataFrame(new_df)
    new_df["vibe_question_types"] = new_df.apply(
        lambda row: get_llm_output(prompt.format(input_text=row["input_text"]), config.proposer.model),
        axis=1,
    )
    return new_df


@timeit
def vibe_discovery(
    df: pd.DataFrame, config: OmegaConf, output_dir: str, current_vibes: List[str] = []
):
    """
    Propose new vibe axes (behaviors) and create preference distribution plot.

    Returns:
        dict: Contains vibes_df and preference distribution plot
    """
    from components.propose import VibeProposer

    models = list(config.models)
    # Create preference distribution plot
    pref_dist = df["preference"].value_counts()
    pref_dist_plot = go.Figure(
        data=[go.Bar(x=pref_dist.index, y=pref_dist.values, marker_color="#2ecc71")]
    )
    pref_dist_plot.update_layout(
        title="Model Preference Distribution",
        xaxis_title="Model",
        yaxis_title="Count",
        template="plotly_white",
    )

    # Log and save preference distribution
    wandb.log({"preference_distribution": wandb.Html(pref_dist_plot.to_html())})
    pref_dist_plot.write_html(os.path.join(output_dir, "preference_distribution.html"))

    # Propose vibes
    vibes = VibeProposer(
        models,
        config,
    ).propose(df.sample(config["proposer"].num_samples, random_state=42).reset_index(drop=True), 
              current_vibes=current_vibes, 
              num_vibes=config.num_vibes)
    print("Proposed Vibes:")
    print("* " + "\n* ".join(vibes))
    print("--------------------------------")

    vibes_df = pd.DataFrame({"vibes": vibes})
    wandb.log({"vibes": wandb.Table(dataframe=vibes_df)})
    vibes_df.to_csv(os.path.join(output_dir, "vibes.csv"), index=False)
    return {"vibes": vibes, "pref_dist_plot": pref_dist_plot}


@timeit
def vibe_validation(
    vibes: List[str], 
    df: pd.DataFrame, 
    config: OmegaConf, 
    # cache: Cache, 
    output_dir: str,
):
    """
    Rank the vibe axes and create visualization plots.

    Returns:
        dict: Contains vibe_df, agg_df, and plots
    """
    from utils import (
        get_pref_score,
        create_side_by_side_plot,
    )
    from components.rank import VibeRankerEmbedding, VibeRanker

    models = list(config.models)

    # Rank vibes (list of strings)
    vibes_to_rank = vibes[:3] if config.test else vibes

    if config.ranker.embedding_rank:
        print("Using embedding ranker")
        vibe_ranker = VibeRankerEmbedding(config)
        vibe_df = vibe_ranker.score(
            vibes_to_rank,
            df,
            single_position_rank=True,
        )
    else:
        vibe_ranker = VibeRanker(config)
        vibe_df = vibe_ranker.score(
            vibes_to_rank,
            df,
            single_position_rank=config.ranker.single_position_rank,
        )

    # Compute preference alignment
    vibe_df["preference_feature"] = vibe_df["preference"].apply(
        lambda x: get_pref_score(x, models)
    )
    vibe_df["pref_score"] = vibe_df["score"] * vibe_df["preference_feature"]

    wandb.log({"Vibe Scoring/ranker_results": wandb.Table(dataframe=vibe_df)})
    vibe_df.to_csv(os.path.join(output_dir, "vibe_df.csv"), index=False)
    agg_df = (
        vibe_df.groupby("vibe")
        .agg({"pref_score": "mean", "score": "mean"})
        .reset_index()
    )
    wandb.log({"summary": wandb.Table(dataframe=agg_df)})

    model_vibe_scores_plot = create_side_by_side_plot(
        df=agg_df,
        y_col="vibe",
        x_cols=["score", "pref_score"],
        titles=("Model Identity", "Preference Prediction"),
        main_title="Vibe Heuristics",
        models=models,
    )
    wandb.log(
        {
            "Vibe Plots/model_vibe_scores_plot": wandb.Html(
                model_vibe_scores_plot.to_html()
            )
        }
    )
    model_vibe_scores_plot.write_html(
        os.path.join(output_dir, "model_vibe_scores_plot.html")
    )

    # Filter vibes with low model differentiation or preference alignment
    filtered_vibes = agg_df[(agg_df["score"].abs() > config.filter.min_score_diff) & (agg_df["pref_score"].abs() > config.filter.min_pref_score_diff)]["vibe"].tolist()
    filtered_vibe_df = vibe_df[vibe_df["vibe"].isin(filtered_vibes)]
    if len(filtered_vibes) < len(agg_df):
        print("Removed vibes for low model differentiation or preference alignment:")
        for vibe in set(agg_df['vibe'].tolist()) - set(filtered_vibes):
            print(f"* {vibe}")
    print("Remaining vibes:")
    print("* " + "\n* ".join(filtered_vibe_df["vibe"].unique()))
    print("--------------------------------")

    return {
        "vibe_df": filtered_vibe_df,
        "agg_df": agg_df,
        "model_vibe_scores_plot": model_vibe_scores_plot,
    }


@timeit
def train_preference_prediction(
    vibe_df: pd.DataFrame, config: OmegaConf, output_dir: str, models: List[str]
):
    """
    Train models and create all analysis plots.

    Returns:
        dict: Contains models, plots, and analysis results
    """
    from utils import (
        train_and_evaluate_model,
        create_side_by_side_plot,
        create_vibe_correlation_plot,
    )

    preference_results = train_and_evaluate_model(
        vibe_df,
        models,
        "preference",
        split_train_test=not config.no_holdout_set,
        solver=config.ranker.solver,
    )
    (
        preference_model,
        preference_coef_df,
        preference_accuracy_test,
        preference_acc_std,
        preference_avg_correct,
    ) = preference_results
    identity_results = train_and_evaluate_model(
        vibe_df,
        models,
        "identity",
        split_train_test=not config.no_holdout_set,
        solver=config.ranker.solver,
    )
    identity_model, identity_coef_df, identity_accuracy_test, identity_acc_std, identity_avg_correct = (
        identity_results
    )
    metrics = {
        "preference_model_test_accuracy": preference_accuracy_test,
        "identity_model_test_accuracy": identity_accuracy_test,
        "preference_model_test_accuracy_std": preference_acc_std,
        "identity_model_test_accuracy_std": identity_acc_std,
    }

    coef_df = identity_coef_df.merge(
        preference_coef_df, on="vibe", suffixes=("_modelID", "_preference")
    ).sort_values("coef_preference", ascending=False)
    coef_plot = create_side_by_side_plot(
        df=coef_df,
        y_col="vibe",
        x_cols=["coef_modelID", "coef_preference"],
        titles=("Model Identity", "Preference Prediction"),
        main_title="Vibe Model Coefficients",
        models=models,
        error_cols=["coef_std_modelID", "coef_std_preference"],
    )

    # Create correlation plot (how much the scores overlap)
    corr_plot = create_vibe_correlation_plot(vibe_df, models)
    coef_plot.write_html(os.path.join(output_dir, "model_vibe_coef_plot.html"))
    coef_df.to_csv(os.path.join(output_dir, "vibecheck_coefficients.csv"), index=False)
    corr_plot.write_html(os.path.join(output_dir, "vibe_correlations.html"))

    df = vibe_df.drop_duplicates("conversation_id").copy()
    df.loc[:, "preference_prediction"] = preference_avg_correct
    df.loc[:, "identity_prediction"] = identity_avg_correct
    # average the preference and identity predictions to avg_prediction
    df.loc[:, "avg_prediction"] = (df["preference_prediction"] + df["identity_prediction"]) / 2

    wandb.log(
        {
            "Vibe Plots/model_vibe_coef_plot": wandb.Html(coef_plot.to_html()),
            "coefficient_data": wandb.Table(dataframe=coef_df),
            "Vibe Scoring/vibe_correlations": wandb.Html(corr_plot.to_html()),
            "Vibe Scoring/df_answers": wandb.Table(dataframe=df),
        }
    )

    return {
        "preference_model": preference_model,
        "identity_model": identity_model,
        "coef_df": coef_df,
        "coef_plot": coef_plot,
        "corr_plot": corr_plot,
        "metrics": metrics,
        "df_answers": df,
    }


def main(config):
    """
    Run VibeCheck analysis pipeline to identify and analyze behavioral differences
    between two language models.

    Args:
        config: OmegaConf configuration object containing all parameters
        
    Returns:
        dict: Results of the analysis including plots and data
    """
    import wandb
    import os
    import pandas as pd

    models = list(config.models)
    # Initialize wandb
    if config.wandb:
        wandb.init(
            project=config.project_name,
            name=f"{models[0]}_vs_{models[1]}",
            save_code=True,
        )

    # Setup output directory
    output_dir = f"{config.output_dir}/{config.data_path.split('/')[-1].replace('.csv', '')}_{models[0]}_vs_{models[1]}"
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    df = pd.read_csv(config.data_path)
    if config.test:
        df = df.sample(min(100, len(df)), random_state=42)
    if "preference" not in df.columns:
        raise ValueError(
            "Preference column not found in dataframe. Run get_preference_labels.py first"
        )
    if not all([c in df.columns for c in models + ["question"]]):
        raise ValueError(
            f"Models {models} or question column not found in dataframe."
        )
    if config.num_samples is not None:
        df = df.sample(config.num_samples, random_state=42)

    df = df[df["preference"].isin(models)].reset_index(drop=True)
    # set conversation_id to be the index of the question, models[0], models[1]
    df["conversation_id"] = df.index
    # Log dataset info
    if config.wandb:
        wandb.summary.update(
            {
                "preference_counts": df["preference"].value_counts().to_dict(),
                "data_size": len(df),
            }
        )

    df["model_a_embedding"] = get_llm_embedding(df[models[0]].tolist(), config.ranker.embedding_model)
    df["model_b_embedding"] = get_llm_embedding(df[models[1]].tolist(), config.ranker.embedding_model)
    df["model_a_embedding"] = df["model_a_embedding"].apply(lambda x: x / np.linalg.norm(x))
    df["model_b_embedding"] = df["model_b_embedding"].apply(lambda x: x / np.linalg.norm(x))
    df = df[df["model_a_embedding"].notna() & df["model_b_embedding"].notna()]

    # # Replace the existing embedding classifier code with:
    classifier_results = train_embedding_classifier(df)
    if config.wandb:
        wandb.log({
            "model_id_embedding_classifier/train_accuracy": classifier_results["train_accuracy"],
            "model_id_embedding_classifier/test_accuracy": classifier_results["test_accuracy"],
            "model_id_embedding_classifier/train_ci": str(classifier_results["train_ci"]),
            "model_id_embedding_classifier/test_ci": str(classifier_results["test_ci"])
        })

    running_vibes = config.initial_vibes
    running_vibe_df = None  # all vibe scores for all iterations
    if len(config.initial_vibes) > 0:
        running_vibes = list(config.initial_vibes)
    vibes_each_iteration = []
    proposer_df = df.sample(config["proposer"].num_samples, random_state=42).reset_index(drop=True)

    for iteration in range(config.iterations):
        # 1. Propose vibes
        propose_results = vibe_discovery(proposer_df, config, output_dir, running_vibes)
        if config.proposer_only:
            if config.wandb:
                wandb.finish()
            return
        running_vibes.extend(list(propose_results["vibes"]))

        # 2. Rank vibes
        rank_results = vibe_validation(
            running_vibes, df, config, output_dir
        )
        rank_results["vibe_df"]["iteration"] = iteration
        if running_vibe_df is None:
            running_vibe_df = rank_results["vibe_df"]
        else:
            running_vibe_df = pd.concat([running_vibe_df, rank_results["vibe_df"]])

        add_running_vibe_df = (
            running_vibe_df.groupby("vibe")
            .agg({"pref_score": "mean", "score": "mean"})
            .reset_index()
        )
        top_vibes = add_running_vibe_df.sort_values("score", ascending=False).head(config.num_vibes) if config.num_vibes is not None else add_running_vibe_df
        ranking_df_iteration = running_vibe_df[running_vibe_df["vibe"].isin(top_vibes["vibe"])]
        running_vibes = running_vibe_df["vibe"].unique().tolist()

        vibes_each_iteration += [{
            "iteration": iteration,
            "vibes": '\n'.join(running_vibe_df["vibe"].unique()),
            "kept_vibes": '\n'.join(top_vibes["vibe"].unique())
        }]

        # 3. Train preference prediction
        train_results = train_preference_prediction(
            ranking_df_iteration, config, output_dir, models
        )
        if config.wandb:
            wandb.log({"iteration": iteration, **train_results["metrics"]})
            wandb.log({"vibes_each_iteration": wandb.Table(dataframe=pd.DataFrame(vibes_each_iteration))})
        proposer_df = train_results["df_answers"].sort_values("preference_prediction", ascending=True)[:config["proposer"].num_samples]

    # 4. Get vibe question types (what types of questions result in high scores for a given vibe)
    vibe_question_types = get_vibe_question_types(rank_results["vibe_df"], config)
    if config.wandb:
        wandb.log(
            {"Vibe Scoring/vibe_question_types": wandb.Table(dataframe=vibe_question_types)}
        )
    vibe_question_types.to_csv(
        os.path.join(output_dir, "vibe_question_types.csv"), index=False
    )

    if config.wandb:
        wandb.log({"vibes_each_iteration": wandb.Table(dataframe=pd.DataFrame(vibes_each_iteration))})
        # Add this line to get the wandb run URL
        wandb_run_url = wandb.run.get_url()
        wandb.finish()
    else:
        wandb_run_url = None

    # Optional: Launch Gradio app
    if config.gradio:
        print("\nLaunching Gradio app...")
        app = create_gradio_app(
            rank_results["vibe_df"],
            models,
            train_results["coef_df"],
            train_results["corr_plot"],
            vibe_question_types,
        )
        app.launch(share=True)

    return {
        "output_dir": output_dir,
        "model_vibe_scores_plot": rank_results["model_vibe_scores_plot"],
        "score_dist_plot": train_results["coef_plot"],
        "vibe_question_types": vibe_question_types,
        "vibe_df": rank_results["vibe_df"],
        "agg_df": rank_results["agg_df"],
        "corr_plot": train_results["corr_plot"],
        "wandb_run_url": wandb_run_url,
        "df": df,  # Return the original dataframe for the webapp
        "models": models,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--webapp", action="store_true", help="Launch the web application")
    known_args, unknown_args = parser.parse_known_args()

    if known_args.webapp:
        from webapp import run_webapp
        run_webapp()
    else:
        # Load base config
        base_config = OmegaConf.load("configs/base.yaml")

        # If --config was passed in, load that as well
        user_config = OmegaConf.load(known_args.config)
        config = OmegaConf.merge(base_config, user_config)

        # Merge with any unknown key=value args from the command line
        # e.g. python main.py data_path=something models=[foo,bar] ...
        cli_config = OmegaConf.from_cli(unknown_args)
        config = OmegaConf.merge(config, cli_config)

        # Now config.* fields are populated from the base config, your custom config,
        # and any command-line overrides.
        if config.data_path is None:
            raise ValueError("data_path must be specified.")
        if config.models is None:
            raise ValueError("models must be specified.")

        if not config.wandb:
            os.environ["WANDB_MODE"] = "offline"

        # Finally, run the main pipeline
        main(config)
