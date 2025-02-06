"""
A Gradio-based UI for running VibeCheck on one or multiple models.
Users can upload a CSV file and select single- or multi-model analysis.
When the user clicks "Run VibeCheck", it calls into the appropriate pipeline,
runs the analysis, and displays the resulting plots, data frames,
and optionally example dropdowns for each discovered vibe.
"""

import os
import tempfile
import gradio as gr
import pandas as pd
from typing import List

# Import your existing code
import messin_around.single_model as single_model
import main as multi_model

# turn wandb off
# os.environ["WANDB_DISABLED"] = "true"


def run_single_vibecheck(
    csv_file: str,
    model_name: str,
    test_mode: bool,
    project_name: str = "vibecheck-single-model",
    proposer_only: bool = False,
    num_proposal_samples: int = 30,
    num_final_vibes: int = 3,
):
    """
    Run the single model vibe check pipeline and return plot/data results.

    Args:
        csv_file (str): Local path to the CSV file.
        model_name (str): The name of the model column in the CSV.
        test_mode (bool): If True, runs the pipeline in test mode (small subset).
        project_name (str): Name for this analysis run
        proposer_only (bool): If True, only run the vibe proposal step
        num_proposal_samples (int): Number of samples to use for vibe proposal
        num_final_vibes (int): Number of top vibes to analyze

    Returns:
        A dictionary of results, containing Plotly figures, vibe_df, etc.
    """
    # Call the single-model pipeline
    result = single_model.main(
        data_path=csv_file,
        model=model_name,
        test=test_mode,
        project_name=project_name,
        proposer_only=proposer_only,
        gradio=False,  # We'll handle final Gradio UI in this file
        num_proposal_samples=num_proposal_samples,
        num_final_vibes=num_final_vibes,
    )
    return result

from omegaconf import OmegaConf
def run_multi_vibecheck(
    csv_file: str,
    models: List[str],
    test_mode: bool,
    project_name: str = "vibecheck",
    proposer_only: bool = False,
    single_position_rank: bool = False,
    no_holdout_set: bool = False,
    num_proposal_samples: int = 30,
    num_final_vibes: int = 10,
):
    """
    Run the multi-model vibe check pipeline and return a dictionary of results.

    Args:
        csv_file (str): Local path to the CSV file.
        models (List[str]): List of model column names in the CSV.
        test_mode (bool): If True, runs the pipeline in test mode (small subset).
        project_name (str): Name for this analysis run
        proposer_only (bool): If True, only run the vibe proposal step
        single_position_rank (bool): Use single position ranking
        no_holdout_set (bool): Don't use a holdout set
        num_proposal_samples (int): Number of samples to use for vibe proposal
        num_final_vibes (int): Number of top vibes to analyze

    Returns:
        A dictionary of results containing plots, dataframes, and analysis results.
    """
    base_config = OmegaConf.load("configs/base.yaml")
    base_config.models = models
    base_config.proposer.num_samples = num_proposal_samples
    base_config.num_final_vibes = num_final_vibes
    base_config.data_path = csv_file
    base_config.test = test_mode
    base_config.project_name = project_name
    base_config.proposer_only = proposer_only
    base_config.ranker.single_position_rank = single_position_rank
    base_config.ranker.no_holdout_set = no_holdout_set
    print(base_config)

    # Call the multi-model pipeline
    result = multi_model.main(base_config)
    # result = multi_model.main(
    #     data_path=csv_file,
    #     models=models,
    #     test=test_mode,
    #     project_name=project_name,
    #     proposer_only=proposer_only,
    #     single_position_rank=single_position_rank,
    #     no_holdout_set=no_holdout_set,
    #     gradio=False,  # We'll handle Gradio UI in this file
    #     num_proposal_samples=num_proposal_samples,
    #     num_final_vibes=num_final_vibes,
    # )

    return result


def create_vibecheck_ui():
    """
    Builds and returns a Gradio Blocks interface which allows users to:
    1. Upload a CSV file,
    2. Choose single- or multi-model analysis,
    3. Specify model name(s),
    4. Optionally set test mode,
    5. Click "Run VibeCheck" to see results,
    6. (For the single-model tab) select discovered vibes to see example prompts.
    """

    with gr.Blocks() as demo:
        gr.Markdown("# <center>It's all about the ✨vibes✨</center>")

        ################################################################
        # SINGLE MODEL TAB
        ################################################################
        with gr.Tab("Single Model (experimental)"):
            gr.Markdown(
                "Upload a CSV containing columns: question, [model_name], preference. "
                "Then specify the model name (the same as the column header for that model)."
            )

            with gr.Row():
                # Left Column - Inputs
                with gr.Column(scale=1):
                    single_csv_file = gr.File(
                        label="Upload CSV for Single Model", file_types=[".csv"]
                    )
                    gr.Examples(
                        examples=["data/friendly_and_cold_sample.csv", "data/cnndm_with_pref.csv"],
                        inputs=single_csv_file,
                        label="Example CSV",
                    )
                    single_csv_columns = gr.Markdown(label="CSV Columns")
                    single_model_name = gr.Textbox(
                        label="Model name (should be one of the columns in the CSV)",
                        placeholder="e.g., gpt4",
                    )
                    single_test_mode = gr.Checkbox(
                        label="Test mode (samples only 100 rows)?", value=False
                    )

                    # Add accordion for advanced parameters
                    with gr.Accordion("Advanced Parameters", open=False):
                        single_project_name = gr.Textbox(
                            label="Project Name",
                            value="vibecheck-single-model",
                            placeholder="Name for this analysis run",
                        )
                        single_proposer_only = gr.Checkbox(
                            label="Proposer Only Mode",
                            value=False,
                            info="Only run the vibe proposal step",
                        )
                        single_num_proposal_samples = gr.Number(
                            label="Number of Proposal Samples",
                            value=30,
                            minimum=1,
                            maximum=100,
                            step=1,
                            info="Number of samples to use for vibe proposal",
                        )
                        single_num_final_vibes = gr.Number(
                            label="Number of Final Vibes",
                            value=3,
                            minimum=1,
                            maximum=20,
                            step=1,
                            info="Number of top vibes to analyze",
                        )

                    run_single_btn = gr.Button("✨VibeCheck✨")
                    single_output_md = gr.Markdown()

                # Right Column - Plots
                with gr.Column(scale=2):
                    single_output_plot1 = gr.Plot()
                    single_output_plot2 = gr.Plot()
                    # single_output_plot_corr = gr.Plot()

            # Bottom Row - Vibe Examples
            with gr.Row():
                with gr.Column():
                    vibe_dropdown = gr.Dropdown(
                        label="Select a vibe for examples",
                        choices=[],
                        interactive=True,
                        multiselect=False,
                        value=None,
                        allow_custom_value=False,
                    )
                    examples_md = gr.Markdown()

            # Store results state
            single_results_state = gr.State()

            def on_run_single(
                csv_file,
                model_name,
                test_mode,
                project_name,
                proposer_only,
                num_proposal_samples,
                num_final_vibes,
            ):
                if not csv_file:
                    return (
                        "Error: No CSV file uploaded.",
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=None),
                        {},
                    )

                # Path to the CSV
                csv_path = csv_file.name
                # Call the single-model pipeline with additional parameters
                results = run_single_vibecheck(
                    csv_path,
                    model_name,
                    test_mode,
                    project_name=project_name,
                    proposer_only=proposer_only,
                    num_proposal_samples=num_proposal_samples,
                    num_final_vibes=num_final_vibes,
                )

                # If something went wrong
                if not results or "vibe_df" not in results:
                    return (
                        "No results returned from pipeline. Check logs.",
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=None),
                        {},
                    )

                # Summarize
                summary_text = (
                    "## VibeCheck Results\n"
                    f"- Output directory: {results['output_dir']}\n"
                    f"- Found {len(results['vibe_df']['vibe'].unique())} vibe(s)\n"
                    + "\n".join(
                        [
                            f"- {vibe}"
                            for vibe in results["vibe_df"]["vibe"].unique().tolist()
                        ]
                    )
                    + "\n### See Examples Below"
                )

                return (
                    summary_text,
                    results["model_vibe_scores_plot"],
                    results["score_dist_plot"],
                    # results["corr_plot"],
                    results,  # store the dictionary
                )

            run_single_btn.click(
                fn=on_run_single,
                inputs=[
                    single_csv_file,
                    single_model_name,
                    single_test_mode,
                    single_project_name,
                    single_proposer_only,
                    single_num_proposal_samples,
                    single_num_final_vibes,
                ],
                outputs=[
                    single_output_md,
                    single_output_plot1,
                    single_output_plot2,
                    single_results_state,
                ],
            )

            ############################################################################
            # Example Selection: Show examples for each discovered vibe
            ############################################################################
            def update_vibe_dropdown(results_dict: dict):
                if not results_dict or "vibe_df" not in results_dict:
                    # Return a new Dropdown with empty choices
                    return gr.Dropdown(choices=[], interactive=True)

                # Get the list of vibes
                choices = sorted(results_dict["vibe_df"]["vibe"].unique().tolist())
                # Return a new Dropdown with the choices
                return gr.Dropdown(choices=choices, interactive=True)

            single_results_state.change(
                fn=update_vibe_dropdown,
                inputs=[single_results_state],
                outputs=[vibe_dropdown],
            )

            def show_single_examples(
                selected_vibe: str, results_dict: dict, model_name: str
            ):
                """Display example rows for the selected vibe."""
                if not selected_vibe:
                    return "Please select a vibe to see examples."

                if not results_dict or "vibe_df" not in results_dict:
                    return "No vibe data available. Please run VibeCheck first."

                vibe_df = results_dict["vibe_df"]
                vibe_question_types = results_dict.get("vibe_question_types", None)

                # Filter for the selected vibe
                subset = vibe_df[
                    (vibe_df["vibe"] == selected_vibe) & (vibe_df["score"].abs() > 0.0)
                ].head(5)

                # If we have vibe_question_types, let's locate any relevant text
                vibe_explanation = ""
                if vibe_question_types is not None:
                    # merge or direct index
                    merged = vibe_df.merge(vibe_question_types, on="vibe", how="left")
                    part = merged[merged["vibe"] == selected_vibe]
                    if len(part) > 0:
                        vibe_explanation = part["vibe_question_types"].values[0]

                md = f"### Questions which exhibit the vibe: {selected_vibe}\n\n"
                if vibe_explanation:
                    md += f"Question types for this vibe:\n\n{vibe_explanation}\n\n"
                md += "---\n\n"

                for i, row in enumerate(subset.itertuples(), 1):
                    # turn pandas object into dict
                    row = row._asdict()
                    print(row.keys())
                    # row.attribute => row.question, row.score ...
                    md += f"#### Example {i}\n"
                    md += f"**Prompt:** {row['question']}\n\n"
                    md += f"**Output:**\n{row[model_name]}\n\n"
                    md += f"**Score:** {row['score']}\n\n"
                    md += f"**Judge / ranker raw output:**\n{row['raw_outputranker_output_1']}\n\n"
                    md += "---\n\n"

                return md

            vibe_dropdown.change(
                fn=show_single_examples,
                inputs=[vibe_dropdown, single_results_state, single_model_name],
                outputs=[examples_md],
            )

            # Update the file handler to show columns
            def on_file_upload(file):
                if file is None:
                    return "No file uploaded."
                if isinstance(file, str):  # Handle example files
                    df = pd.read_csv(file)
                else:  # Handle uploaded files
                    df = pd.read_csv(file.name)
                columns = df.columns.tolist()
                return f"### CSV Columns:\n" + "\n".join(
                    [f"- {col}" for col in columns]
                )

            # Add the file upload and example handlers for single model
            single_csv_file.upload(
                fn=on_file_upload,
                inputs=[single_csv_file],
                outputs=[single_csv_columns],
            )
            single_csv_file.change(  # Add handler for example clicks
                fn=on_file_upload,
                inputs=[single_csv_file],
                outputs=[single_csv_columns],
            )

        ################################################################
        # MULTI MODEL TAB
        ################################################################
        with gr.Tab("Compare Two Models"):
            gr.Markdown(
                "Data should contain columns: question, model1, model2, preference"
            )

            with gr.Row():
                # Left Column - Inputs
                with gr.Column(scale=1):
                    multi_csv_file = gr.File(
                        label="Upload CSV for Multi Model", file_types=[".csv"]
                    )
                    gr.Examples(
                        examples=["data/friendly_and_cold_sample.csv", "data/cnndm_with_pref.csv"],
                        inputs=multi_csv_file,
                        label="Example CSV",
                    )
                    multi_csv_columns = gr.Markdown(label="CSV Columns")
                    multi_models = gr.Textbox(
                        label="Enter model column names (comma-separated)",
                        placeholder="modelA, modelB",
                    )
                    multi_test_mode = gr.Checkbox(
                        label="Test mode? (sample 100 rows)", value=False
                    )

                    # Add accordion for advanced parameters
                    with gr.Accordion("Advanced Parameters", open=False):
                        multi_project_name = gr.Textbox(
                            label="Project Name",
                            value="vibecheck",
                            placeholder="Name for this analysis run",
                        )
                        multi_proposer_only = gr.Checkbox(
                            label="Proposer Only Mode",
                            value=False,
                            info="Only run the vibe proposal step",
                        )
                        multi_single_position_rank = gr.Checkbox(
                            label="Single Position Rank",
                            value=False,
                            info="Use single position ranking",
                        )
                        multi_no_holdout_set = gr.Checkbox(
                            label="No Holdout Set",
                            value=False,
                            info="Don't use a holdout set",
                        )
                        multi_num_proposal_samples = gr.Number(
                            label="Number of Proposal Samples",
                            value=30,
                            minimum=1,
                            maximum=100,
                            step=1,
                            info="Number of samples to use for vibe proposal",
                        )
                        multi_num_final_vibes = gr.Number(
                            label="Number of Final Vibes",
                            value=10,
                            minimum=1,
                            maximum=20,
                            step=1,
                            info="Number of top vibes to analyze",
                        )

                    run_multi_btn = gr.Button("✨VibeCheck✨")

                # Right Column - Plots
                with gr.Column(scale=2):
                    multi_output_plot1 = gr.Plot()
                    multi_output_plot2 = gr.Plot()

            with gr.Row():
                multi_output_md = gr.Markdown()

            # Bottom Row - Vibe Examples
            with gr.Row():
                with gr.Column():
                    multi_vibe_dropdown = gr.Dropdown(
                        label="Select a vibe for examples",
                        choices=[],
                        interactive=True,
                        multiselect=False,
                        value=None,
                        allow_custom_value=False,
                    )
                    multi_examples_md = gr.Markdown()

            # Store results state
            multi_results_state = gr.State()

            def on_run_multi(
                csv_file,
                model_list_str,
                test_mode,
                project_name,
                proposer_only,
                single_position_rank,
                no_holdout_set,
                num_proposal_samples,
                num_final_vibes,
            ):
                if not csv_file:
                    return (
                        "Error: No CSV uploaded.",
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=None),
                        {},
                    )

                csv_path = csv_file.name
                model_list = [m.strip() for m in model_list_str.split(",") if m.strip()]

                # Call multi-model pipeline with additional parameters
                results = run_multi_vibecheck(
                    csv_path,
                    model_list,
                    test_mode,
                    project_name=project_name,
                    proposer_only=proposer_only,
                    single_position_rank=single_position_rank,
                    no_holdout_set=no_holdout_set,
                    num_proposal_samples=num_proposal_samples,
                    num_final_vibes=num_final_vibes,
                )

                # If something went wrong
                if not results or "vibe_df" not in results:
                    return (
                        "No results returned from pipeline. Check logs.",
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=None),
                        {},
                    )

                # Summarize
                summary_text = (
                    "## Multi-Model VibeCheck Results\n"
                    + f"Output directory: {results['output_dir']}\n\n"
                    + f"Wandb Run URL: {results['wandb_run_url']}\n\n"
                    + f"**Found {len(results['vibe_df']['vibe'].unique())} vibe(s)**\n"
                    + "\n".join(
                        [
                            f"- {vibe}"
                            for vibe in results["vibe_df"]["vibe"].unique().tolist()
                        ]
                    )
                    + "\n### See Examples Below"
                )

                return (
                    summary_text,
                    results["model_vibe_scores_plot"],
                    results["score_dist_plot"],
                    # results["corr_plot"],
                    results,  # store the dictionary
                )

            run_multi_btn.click(
                fn=on_run_multi,
                inputs=[
                    multi_csv_file,
                    multi_models,
                    multi_test_mode,
                    multi_project_name,
                    multi_proposer_only,
                    multi_single_position_rank,
                    multi_no_holdout_set,
                    multi_num_proposal_samples,
                    multi_num_final_vibes,
                ],
                outputs=[
                    multi_output_md,
                    multi_output_plot1,
                    multi_output_plot2,
                    multi_results_state,
                ],
            )

            ############################################################################
            # Example Selection: Show examples for each discovered vibe
            ############################################################################
            def update_multi_vibe_dropdown(results_dict: dict):
                if not results_dict or "vibe_df" not in results_dict:
                    return gr.Dropdown(choices=[], interactive=True)

                choices = sorted(results_dict["vibe_df"]["vibe"].unique().tolist())
                return gr.Dropdown(choices=choices, interactive=True)

            multi_results_state.change(
                fn=update_multi_vibe_dropdown,
                inputs=[multi_results_state],
                outputs=[multi_vibe_dropdown],
            )

            def show_multi_examples(
                selected_vibe: str, results_dict: dict, model_list_str: str
            ):
                """Display example rows for the selected vibe."""
                if not selected_vibe:
                    return "Please select a vibe to see examples."

                if not results_dict or "vibe_df" not in results_dict:
                    return "No vibe data available. Please run VibeCheck first."

                models = [m.strip() for m in model_list_str.split(",") if m.strip()]
                if len(models) != 2:
                    return "Please specify exactly two models."

                vibe_df = results_dict["vibe_df"]
                vibe_question_types = results_dict.get("vibe_question_types", None)

                # Filter for the selected vibe
                subset = vibe_df[
                    (vibe_df["vibe"] == selected_vibe) & (vibe_df["score"].abs() > 0.0)
                ].head(5)

                # If we have vibe_question_types, let's locate any relevant text
                vibe_explanation = ""
                if vibe_question_types is not None:
                    merged = vibe_df.merge(vibe_question_types, on="vibe", how="left")
                    part = merged[merged["vibe"] == selected_vibe]
                    if len(part) > 0:
                        vibe_explanation = part["vibe_question_types"].values[0]

                md = f"### Questions which exhibit the vibe: {selected_vibe}\n\n"
                if vibe_explanation:
                    md += f"Question types for this vibe:\n\n{vibe_explanation}\n\n"
                md += "---\n\n"

                for i, row in enumerate(subset.itertuples(), 1):
                    row = row._asdict()
                    md += f"#### Example {i}\n"
                    md += f"**Prompt:** {row['question']}\n\n"
                    # Show both model outputs side by side
                    md += f"**{models[0]} Output:**\n{row[models[0]]}\n\n"
                    md += f"**{models[1]} Output:**\n{row[models[1]]}\n\n"
                    md += f"**Score:** {row['score']:.3f} "
                    # Add interpretation of which model exhibits the vibe more
                    if row["score"] > 0:
                        md += f"({models[0]} exhibits this vibe more)\n\n"
                    else:
                        md += f"({models[1]} exhibits this vibe more)\n\n"
                    md += f"**Judge / ranker raw output:**\n{row['raw_outputranker_output_1']}\n\n"
                    md += "---\n\n"

                return md

            multi_vibe_dropdown.change(
                fn=show_multi_examples,
                inputs=[multi_vibe_dropdown, multi_results_state, multi_models],
                outputs=[multi_examples_md],
            )

            # Add the file upload and example handlers for multi-model
            multi_csv_file.upload(
                fn=on_file_upload, inputs=[multi_csv_file], outputs=[multi_csv_columns]
            )
            multi_csv_file.change(  # Add handler for example clicks
                fn=on_file_upload, inputs=[multi_csv_file], outputs=[multi_csv_columns]
            )

    return demo


os.environ["GRADIO_TEMP_DIR"] = "./gradio_cache"

if __name__ == "__main__":
    app = create_vibecheck_ui()
    # To share publicly on a unique link, set share=True
    app.launch(share=True)
