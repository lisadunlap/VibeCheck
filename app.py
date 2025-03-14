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
    config_overrides: dict = None,
):
    """
    Run the multi-model vibe check pipeline and return a dictionary of results.

    Args:
        csv_file (str): Local path to the CSV file.
        models (List[str]): List of model column names in the CSV.
        test_mode (bool): If True, runs the pipeline in test mode (small subset).
        config_overrides (dict): Dictionary of configuration overrides.

    Returns:
        A dictionary of results containing plots, dataframes, and analysis results.
    """
    base_config = OmegaConf.load("configs/base.yaml")
    base_config.models = models
    base_config.data_path = csv_file
    base_config.test = test_mode
    
    # Apply any configuration overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if value is not None:  # Only override if value is provided
                # Handle nested keys like "proposer.num_samples"
                if "." in key:
                    parts = key.split(".")
                    current = base_config
                    for part in parts[:-1]:
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    base_config[key] = value
    
    print(base_config)

    # Call the multi-model pipeline
    result = multi_model.main(base_config)

    return result

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
    
    # Custom CSS for a more professional look
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 1rem;
    }
    .header-emoji {
        font-size: 2.5rem;
        margin-right: 0.5rem;
    }
    .tab-content {
        padding: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        gr.HTML("""
        <div class="header-text">
            <h1><span class="header-emoji">✨</span> VibeCheck <span class="header-emoji">✨</span></h1>
            <p>Discover and analyze the vibes in your model outputs</p>
        </div>
        """)

        ################################################################
        # MULTI MODEL TAB
        ################################################################
        with gr.Tab("Compare Two Models"):
            gr.HTML("""
            <div class="tab-content">
                <h3>Model Comparison Analysis</h3>
                <p>Upload a CSV file containing model outputs to compare and discover the vibes that differentiate them.</p>
                <p><b>Required CSV format:</b> Your data should contain columns for <code>question</code>, model outputs (e.g., <code>model1</code>, <code>model2</code>), and <code>preference</code>.</p>
            </div>
            """)

            with gr.Row():
                # Left Column - Inputs
                with gr.Column(scale=1):
                    multi_csv_file = gr.File(
                        label="Upload CSV for Model Comparison", 
                        file_types=[".csv"],
                        elem_id="multi-csv-upload"
                    )
                    gr.Examples(
                        examples=["data/friendly_and_cold_sample.csv", "data/cnndm_with_pref.csv"],
                        inputs=multi_csv_file,
                        label="Example Datasets",
                    )
                    multi_csv_columns = gr.Markdown(label="CSV Columns")
                    multi_models = gr.Textbox(
                        label="Model Column Names (comma-separated with a space between each model name)",
                        placeholder="modelA, modelB",
                        elem_id="multi-model-names"
                    )
                    multi_test_mode = gr.Checkbox(
                        label="Test Mode (sample 100 rows for faster results)", 
                        value=False
                    )

                    # Add accordion for advanced parameters
                    with gr.Accordion("Advanced Parameters", open=False):
                        # Load the base config to get default values
                        base_config = OmegaConf.load("configs/base.yaml")
                        config_inputs = {}
                        
                        # General Settings
                        with gr.Accordion("General Settings", open=True):
                            config_inputs["project_name"] = gr.Textbox(
                                label="Project Name",
                                value=base_config.project_name,
                                info="Name of the Weights & Biases project where results will be logged."
                            )
                            
                            config_inputs["wandb"] = gr.Checkbox(
                                label="Log to Weights & Biases",
                                value=base_config.wandb,
                                info="If true, logs results to Weights & Biases."
                            )
                            
                            config_inputs["output_dir"] = gr.Textbox(
                                label="Output Directory",
                                value=base_config.output_dir,
                                info="Directory to save results."
                            )
                            
                            config_inputs["proposer_only"] = gr.Checkbox(
                                label="Proposer Only Mode",
                                value=base_config.proposer_only,
                                info="If true, only runs the vibe proposal step without conducting further analysis."
                            )
                            
                            config_inputs["no_holdout_set"] = gr.Checkbox(
                                label="No Holdout Set",
                                value=base_config.no_holdout_set,
                                info="If true, uses all data for training without creating a separate test set."
                            )
                            
                            config_inputs["gradio"] = gr.Checkbox(
                                label="Launch Gradio Interface",
                                value=base_config.gradio,
                                info="Launch a Gradio interface after analysis for interactive exploration of the results."
                            )
                            
                            config_inputs["num_vibes"] = gr.Number(
                                label="Maximum Number of Vibes",
                                value=base_config.num_vibes,
                                minimum=1,
                                step=1,
                                info="Maximum number of vibes to use in final analysis."
                            )
                            
                            config_inputs["iterations"] = gr.Number(
                                label="Number of Iterations",
                                value=base_config.iterations,
                                minimum=1,
                                step=1,
                                info="Number of iterations to run the analysis."
                            )
                        
                        # Proposer Settings
                        with gr.Accordion("Proposer Settings", open=False):
                            config_inputs["proposer.num_samples"] = gr.Number(
                                label="Number of Proposal Samples",
                                value=base_config.proposer.num_samples,
                                minimum=1,
                                step=1,
                                info="Number of samples to use when proposing vibes."
                            )
                            
                            config_inputs["proposer.model"] = gr.Textbox(
                                label="Proposer Model",
                                value=base_config.proposer.model,
                                info="Model to use for vibe proposal."
                            )
                            
                            config_inputs["proposer.shuffle_positions"] = gr.Checkbox(
                                label="Shuffle Positions",
                                value=base_config.proposer.shuffle_positions,
                                info="If true, shuffles the positions of the models in the prompt for finding vibes."
                            )
                            
                            config_inputs["proposer.batch_size"] = gr.Number(
                                label="Batch Size",
                                value=base_config.proposer.batch_size,
                                minimum=1,
                                step=1,
                                info="Number of samples to use for each batch in the vibe proposal."
                            )
                            
                            config_inputs["proposer.embedding_model"] = gr.Textbox(
                                label="Embedding Model",
                                value=base_config.proposer.embedding_model,
                                info="Model to use for embedding analysis."
                            )
                            
                            config_inputs["proposer.prompt"] = gr.Textbox(
                                label="Proposer Prompt",
                                value=base_config.proposer.prompt,
                                info="Type of prompt to use for vibe proposal."
                            )
                            
                            config_inputs["proposer.iteration_prompt"] = gr.Textbox(
                                label="Iteration Prompt",
                                value=base_config.proposer.iteration_prompt,
                                info="Type of prompt to use for vibe proposal iteration."
                            )
                            
                            config_inputs["proposer.reduction_prompt"] = gr.Textbox(
                                label="Reduction Prompt",
                                value=base_config.proposer.reduction_prompt,
                                info="Type of prompt to use for axis reduction."
                            )
                        
                        # Ranker Settings
                        with gr.Accordion("Ranker Settings", open=False):
                            config_inputs["ranker.single_position_rank"] = gr.Checkbox(
                                label="Single Position Rank",
                                value=base_config.ranker.single_position_rank,
                                info="If true, only ranks model outputs in one position order."
                            )
                            
                            config_inputs["ranker.model"] = gr.Textbox(
                                label="Ranker Model",
                                value=base_config.ranker.model,
                                info="Model to use for ranking."
                            )
                            
                            config_inputs["ranker.solver"] = gr.Dropdown(
                                label="Solver",
                                choices=["standard", "lasso", "elasticnet"],
                                value=base_config.ranker.solver,
                                info="Solver to use for regression analysis."
                            )
                            
                            config_inputs["ranker.num_final_vibes"] = gr.Number(
                                label="Number of Final Vibes",
                                value=base_config.ranker.num_final_vibes,
                                minimum=1,
                                step=1,
                                info="Maximum number of vibes to use in final analysis."
                            )
                            
                            config_inputs["ranker.embedding_model"] = gr.Textbox(
                                label="Embedding Model",
                                value=base_config.ranker.embedding_model,
                                info="Model to use for embedding analysis."
                            )
                            
                            config_inputs["ranker.embedding_rank"] = gr.Checkbox(
                                label="Embedding Rank",
                                value=base_config.ranker.embedding_rank,
                                info="If true, ranks model outputs using embedding similarity."
                            )
                            
                            config_inputs["ranker.vibe_batch_size"] = gr.Number(
                                label="Vibe Batch Size",
                                value=base_config.ranker.vibe_batch_size,
                                minimum=1,
                                step=1,
                                info="Batch size for ranker."
                            )
                            
                            config_inputs["ranker.prompt"] = gr.Textbox(
                                label="Ranker Prompt",
                                value=base_config.ranker.prompt,
                                info="Type of prompt to use for ranker."
                            )
                        
                        # Filter Settings
                        with gr.Accordion("Filter Settings", open=False):
                            config_inputs["filter.min_score_diff"] = gr.Number(
                                label="Minimum Score Difference",
                                value=base_config.filter.min_score_diff,
                                info="Minimum score difference to consider a vibe."
                            )
                            
                            config_inputs["filter.min_pref_score_diff"] = gr.Number(
                                label="Minimum Preference Score Difference",
                                value=base_config.filter.min_pref_score_diff,
                                info="Minimum preference score difference to consider a vibe."
                            )
                        
                    run_multi_btn = gr.Button("✨ Run VibeCheck ✨", variant="primary", size="lg")

                with gr.Column():
                    multi_output_md = gr.Markdown()

            # Right Column - Plots
            with gr.Row():
                gr.Markdown("### Vibe Score Distribution")
                multi_output_plot1 = gr.Plot()
                
            with gr.Row():
                gr.Markdown("### Model Vibe Scores")
                multi_output_plot2 = gr.Plot()

            # Bottom Row - Vibe Examples
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Explore Vibe Examples")
                    multi_vibe_dropdown = gr.Dropdown(
                        label="Select a vibe to see examples",
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
                *args
            ):
                if not csv_file:
                    return (
                        "Error: No CSV uploaded.",
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=None),
                    )

                csv_path = csv_file.name
                model_list = [m.strip() for m in model_list_str.split(",") if m.strip()]
                
                # Collect all configuration overrides
                config_overrides = {}
                for i, key in enumerate(config_inputs.keys()):
                    if i < len(args) and args[i] is not None:  # Only include non-None values
                        config_overrides[key] = args[i]

                # Call multi-model pipeline with configuration overrides
                results = run_multi_vibecheck(
                    csv_path,
                    model_list,
                    test_mode,
                    config_overrides=config_overrides
                )

                # If something went wrong
                if not results or "vibe_df" not in results:
                    return (
                        "No results returned from pipeline. Check logs.",
                        gr.update(value=None),
                        gr.update(value=None),
                        gr.update(value=None),
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

            # Collect all config inputs for the click handler
            config_input_list = list(config_inputs.values())
            config_input_names = list(config_inputs.keys())
            
            run_multi_btn.click(
                fn=on_run_multi,
                inputs=[
                    multi_csv_file,
                    multi_models,
                    multi_test_mode,
                    *config_input_list
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
                    print(subset.columns)
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
                    # md += f"**Judge / ranker raw output:**\n{row['ranker_output']}\n\n"
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

        ################################################################
        # SINGLE MODEL TAB
        ################################################################
        with gr.Tab("Single Model (experimental)"):
            gr.HTML("""
            <div class="tab-content">
                <h3>Single Model Analysis</h3>
                <p>Upload a CSV containing a single model's outputs to discover its characteristic vibes.</p>
                <p><b>Required CSV format:</b> Your data should contain columns for <code>question</code>, your model's outputs (e.g., <code>gpt4</code>), and <code>preference</code>.</p>
            </div>
            """)

            with gr.Row():
                # Left Column - Inputs
                with gr.Column(scale=1):
                    single_csv_file = gr.File(
                        label="Upload CSV for Single Model Analysis", 
                        file_types=[".csv"],
                        elem_id="single-csv-upload"
                    )
                    gr.Examples(
                        examples=["data/friendly_and_cold_sample.csv", "data/cnndm_with_pref.csv"],
                        inputs=single_csv_file,
                        label="Example Datasets",
                    )
                    single_csv_columns = gr.Markdown(label="CSV Columns")
                    single_model_name = gr.Textbox(
                        label="Model Column Name",
                        placeholder="e.g., gpt4",
                        info="Should match one of the column names in your CSV",
                        elem_id="single-model-name"
                    )
                    single_test_mode = gr.Checkbox(
                        label="Test Mode (sample 100 rows for faster results)", 
                        value=False
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

                    run_single_btn = gr.Button("✨ Run VibeCheck ✨", variant="primary", size="lg")
                    single_output_md = gr.Markdown()

                # Right Column - Plots
                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Results")
                    single_output_plot1 = gr.Plot()
                    single_output_plot2 = gr.Plot()

            # Bottom Row - Vibe Examples
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Explore Vibe Examples")
                    vibe_dropdown = gr.Dropdown(
                        label="Select a vibe to see examples",
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
                    md += f"**Judge / ranker raw output:**\n{row['ranker_output_1']}\n\n"
                    md += "---\n\n"

                return md

            vibe_dropdown.change(
                fn=show_single_examples,
                inputs=[vibe_dropdown, single_results_state, single_model_name],
                outputs=[examples_md],
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

        # Add a footer
        gr.HTML("""
        <div class="footer">
            <p>VibeCheck - Discover the vibes in your model outputs</p>
            <p>© 2023 VibeCheck Team</p>
        </div>
        """)

    return demo


os.environ["GRADIO_TEMP_DIR"] = "./gradio_cache"

if __name__ == "__main__":
    app = create_vibecheck_ui()
    # To share publicly on a unique link, set share=True
    app.launch(share=True)
