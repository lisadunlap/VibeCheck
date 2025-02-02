import argparse
import wandb
import pandas as pd
from plotly import graph_objects as go
import numpy as np
import os
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.cache import CacheConfig, CacheType, CacheFactory
from omegaconf import OmegaConf
from lotus.cache import Cache

from typing import List

from utils import (
    create_gradio_app,
)


def get_vibe_question_types(vibe_df: pd.DataFrame, batch_size: int = 50):
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
    vibe_question_types = new_df.sem_map(
        prompt, return_raw_outputs=True, suffix="vibe_question_types"
    )
    return vibe_question_types


def vibe_discovery(
    df: pd.DataFrame, config: OmegaConf, output_dir: str, current_vibes: List[str] = []
):
    """
    Propose new vibe axes (behaviors) and create preference distribution plot.

    Returns:
        dict: Contains vibes_df and preference distribution plot
    """
    from components.propose import propose_vibes

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
    vibes = propose_vibes(
        df,
        config.models,
        num_proposal_samples=config.proposer.num_samples,
        num_final_vibes=config.num_final_vibes,
        current_vibes=current_vibes,
    )

    vibes_df = pd.DataFrame({"vibes": vibes})
    wandb.log({"vibes": wandb.Table(dataframe=vibes_df)})
    vibes_df.to_csv(os.path.join(output_dir, "vibes.csv"), index=False)

    return {"vibes": vibes, "pref_dist_plot": pref_dist_plot}


def vibe_validation(
    vibes: List[str], 
    df: pd.DataFrame, 
    config: OmegaConf, 
    cache: Cache, 
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
    from components.rank import rank_vibes, rank_vibes_embedding

    # Change model for ranking
    lm = LM(model=config.ranker.model, cache=cache)
    lotus.settings.configure(lm=lm, enable_cache=True)

    # Rank vibes (list of strings)
    vibes_to_rank = vibes[:3] if config.test else vibes

    if config.ranker.embedding_rank:
        vibe_df = rank_vibes_embedding(
            vibes_to_rank, 
            df, 
            config.models, 
            config.ranker.embedding_model)
    else:
        vibe_df = rank_vibes(
            vibes_to_rank,
            df,
            config.models,
            single_position_rank=config.ranker.single_position_rank,
        )

    # Compute preference alignment
    vibe_df["preference_feature"] = vibe_df["preference"].apply(
        lambda x: get_pref_score(x, config.models)
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
        models=config.models,
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
    filtered_vibe_df = vibe_df[
        (vibe_df["score"].abs() > 0.05) & (vibe_df["pref_score"].abs() > 0.05)
    ]
    print(
        f"Retained {len(filtered_vibe_df.drop_duplicates('vibe'))} vibes with non-trivial separation/preference."
    )
    print("Remaining vibes:\n" + "\n".join(filtered_vibe_df["vibe"].unique()))

    return {
        "vibe_df": filtered_vibe_df,
        "agg_df": agg_df,
        "model_vibe_scores_plot": model_vibe_scores_plot,
    }


def train_preference_prediction(
    vibe_df: pd.DataFrame, config: OmegaConf, output_dir: str
):
    """
    Train models and create all analysis plots.

    Returns:
        dict: Contains models, plots, and analysis results
    """
    from utils import (
        get_feature_df,
        train_and_evaluate_model,
        create_side_by_side_plot,
        create_vibe_correlation_plot,
    )

    feature_df, X_pref, y_pref, y_identity = get_feature_df(vibe_df)
    preference_results = train_and_evaluate_model(
        X_pref,
        y_pref,
        feature_df.columns,
        split_train_test=not config.no_holdout_set,
        solver=config.ranker.solver,
    )
    (
        preference_model,
        preference_coef_df,
        preference_accuracy_test,
        preference_acc_std,
    ) = preference_results
    identity_results = train_and_evaluate_model(
        X_pref,
        y_identity,
        feature_df.columns,
        split_train_test=not config.no_holdout_set,
        solver=config.ranker.solver,
    )
    identity_model, identity_coef_df, identity_accuracy_test, identity_acc_std = (
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
        models=config.models,
        error_cols=["coef_std_modelID", "coef_std_preference"],
    )

    # Create correlation plot (how much the scores overlap)
    corr_plot = create_vibe_correlation_plot(vibe_df, config.models)

    # Log and save results
    wandb.log(
        {
            "Vibe Plots/model_vibe_coef_plot": wandb.Html(coef_plot.to_html()),
            "coefficient_data": wandb.Table(dataframe=coef_df),
            "Vibe Scoring/vibe_correlations": wandb.Html(corr_plot.to_html()),
        }
    )

    coef_plot.write_html(os.path.join(output_dir, "model_vibe_coef_plot.html"))
    coef_df.to_csv(os.path.join(output_dir, "vibecheck_coefficients.csv"), index=False)
    corr_plot.write_html(os.path.join(output_dir, "vibe_correlations.html"))

    return {
        "preference_model": preference_model,
        "identity_model": identity_model,
        "coef_df": coef_df,
        "coef_plot": coef_plot,
        "corr_plot": corr_plot,
        "metrics": metrics,
    }


def main(config):
    """
    Run VibeCheck analysis pipeline to identify and analyze behavioral differences
    between two language models.

    Args:
        config: OmegaConf configuration object containing all parameters
    """
    import wandb
    import os
    import lotus
    from lotus.models import LM, SentenceTransformersRM
    from lotus.cache import CacheConfig, CacheType, CacheFactory
    import pandas as pd

    # Initialize wandb
    wandb.init(
        project=config.project_name,
        name=f"{config.models[0]}_vs_{config.models[1]}",
        save_code=True,
    )

    # Setup output directory
    output_dir = f"{config.output_dir}/{config.data_path.split('/')[-1].replace('.csv', '')}_{config.models[0]}_vs_{config.models[1]}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize LOTUS
    cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
    cache = CacheFactory.create_cache(cache_config)
    lm = LM(model=config.proposer.model, cache=cache)
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    lotus.settings.configure(lm=lm, rm=rm, enable_cache=True)

    # Load and preprocess data
    df = pd.read_csv(config.data_path)
    if config.test:
        df = df.sample(min(100, len(df)), random_state=42)
    if "preference" not in df.columns:
        raise ValueError(
            "Preference column not found in dataframe. Run get_preference_labels.py first"
        )
    if not all([c in df.columns for c in config.models + ["question"]]):
        raise ValueError(
            f"Models {config.models} or question column not found in dataframe."
        )

    df = df[df["preference"].isin(config.models)].reset_index(drop=True)
    # Log dataset info
    wandb.summary.update(
        {
            "preference_counts": df["preference"].value_counts().to_dict(),
            "data_size": len(df),
        }
    )

    running_vibes = config.initial_vibes
    running_vibe_df = None  # all vibe scores for all iterations
    if len(config.initial_vibes) > 0:
        running_vibes = config.initial_vibes
    vibes_each_iteration = []

    for iteration in range(config.iterations):
        # 1. Propose vibes
        propose_results = vibe_discovery(df, config, output_dir, running_vibes)
        if config.proposer_only:
            wandb.finish()
            return
        running_vibes = running_vibes +propose_results["vibes"]

        # 2. Rank vibes
        rank_results = vibe_validation(
            running_vibes, df, config, cache, output_dir
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
        top_vibes = add_running_vibe_df.sort_values("score", ascending=False).head(config.num_final_vibes)
        ranking_df_iteration = running_vibe_df[running_vibe_df["vibe"].isin(top_vibes["vibe"])]
        running_vibes = running_vibe_df["vibe"].unique()

        vibes_each_iteration += [{
            "iteration": iteration,
            "vibes": running_vibe_df["vibe"].unique(),
            "kept_vibes": top_vibes["vibe"].unique()
        }]

        # 3. Train preference prediction
        train_results = train_preference_prediction(
            ranking_df_iteration, config, output_dir
        )
        wandb.log({"iteration": iteration, **train_results["metrics"]})
        wandb.log({"vibes_each_iteration": wandb.Table(dataframe=pd.DataFrame(vibes_each_iteration))})

    # 4. Get vibe question types (what types of questions result in high scores for a given vibe)
    vibe_question_types = get_vibe_question_types(rank_results["vibe_df"])
    wandb.log(
        {"Vibe Scoring/vibe_question_types": wandb.Table(dataframe=vibe_question_types)}
    )
    vibe_question_types.to_csv(
        os.path.join(output_dir, "vibe_question_types.csv"), index=False
    )

    wandb.log({"vibes_each_iteration": wandb.Table(dataframe=pd.DataFrame(vibes_each_iteration))})

    wandb.finish()

    # Optional: Launch Gradio app
    if config.gradio:
        print("\nLaunching Gradio app...")
        app = create_gradio_app(
            rank_results["vibe_df"],
            config.models,
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
    }


if __name__ == "__main__":
    # Load default config
    config = OmegaConf.load("configs/base.yaml")

    # Merge with CLI arguments if any
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_config)

    # Validate required arguments
    if config.data_path is None:
        raise ValueError("data_path must be specified")
    if config.models is None:
        raise ValueError("models must be specified")

    if not config.wandb:
        os.environ["WANDB_MODE"] = "offline"

    # getting the party started
    main(config)
