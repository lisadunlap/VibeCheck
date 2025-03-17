"""
A Gradio-based UI for running VibeCheck on multiple models.
Users can upload a CSV file and run the analysis.
When the user clicks "Run VibeCheck", it calls into the pipeline,
runs the analysis, and displays the resulting plots, data frames,
and example dropdowns for each discovered vibe.
"""

import os
import gradio as gr
import pandas as pd
from typing import List
from omegaconf import OmegaConf

# Import your existing code
import main as multi_model

# Helper functions
def on_file_upload(file):
    """Parse uploaded CSV file and display column names"""
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

def list_saved_results():
    """List all saved result files in the saved_results directory."""
    results_dir = os.path.join("vibecheck_results")
    if not os.path.exists(results_dir):
        return []
    
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".pkl")]
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    
    # Format the filenames for display
    formatted_results = []
    for filename in result_files:
        # Extract date from filename
        parts = filename.split("_")
        if len(parts) >= 2:
            # Try to make the filename more readable
            try:
                # Assuming format is model1_vs_model2_dataset_YYYYMMDD_HHMMSS.pkl
                models_part = "_".join(parts[:-2])  # Everything except timestamp
                date_part = parts[-2]
                time_part = parts[-1].replace(".pkl", "")
                
                # Format date for display
                date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
                time_str = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
                
                display_name = f"{models_part} ({date_str} {time_str})"
                formatted_results.append((display_name, filename))
            except:
                # If parsing fails, just use the filename
                formatted_results.append((filename, filename))
        else:
            formatted_results.append((filename, filename))
    
    return formatted_results

def load_saved_result(filename):
    """Load a saved result file."""
    import pickle
    
    results_dir = os.path.join("vibecheck_results")
    filepath = os.path.join(results_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, "rb") as f:
        results = pickle.load(f)
    
    print(results['df'].columns)
    print("+++++++++++")
    print("+++++++++++")
    return results

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

    # Call the multi-model pipeline
    result = multi_model.main(base_config)
    
    return result

def show_examples(selected_vibe, results_dict, model_list_str=None):
    """Display example rows for the selected vibe."""
    if not selected_vibe:
        return "Please select a vibe to see examples.", []
    
    if not results_dict or "vibe_df" not in results_dict:
        return "No vibe data available.", []
    
    vibe_df = results_dict["vibe_df"]
    
    # Extract model names from the results dictionary or use provided models
    if model_list_str:
        models = [m.strip() for m in model_list_str.split(",") if m.strip()]
    else:
        models = results_dict.get("models", ["Model A", "Model B"])
    
    # Filter for the selected vibe
    subset = vibe_df[ (vibe_df["vibe"] == selected_vibe)].head(10)
    
    # Create a list of example choices with preview text
    example_choices = []
    for i, row in enumerate(subset.itertuples(), 1):
        row = row._asdict()
        # Create a short preview of the prompt (first 100 chars)
        preview = row['question'][:100] + "..." if len(row['question']) > 100 else row['question']
        score = row['score']
        # Format: "Example 1: This is the prompt... (Score: 0.123)"
        label = f"Example {i}: {preview} (Score: {score:.3f})"
        example_choices.append(label)
    
    return f"Found {len(example_choices)} examples for vibe: {selected_vibe.replace('**', '')}", gr.Dropdown(choices=example_choices, value=example_choices[0] if example_choices else None)

def display_selected_example(example_idx, selected_vibe, results_dict, model_list_str=None):
    """Display a single selected example."""
    if example_idx is None or not selected_vibe:
        return "Please select an example to view."
    
    if not results_dict or "vibe_df" not in results_dict:
        return "No vibe data available."
    
    vibe_df = results_dict["vibe_df"]
    
    # Extract model names from the results dictionary or use provided models
    if model_list_str:
        models = [m.strip() for m in model_list_str.split(",") if m.strip()]
    else:
        models = results_dict.get("models", ["Model A", "Model B"])
    
    # Filter for the selected vibe
    subset = vibe_df[
        (vibe_df["vibe"] == selected_vibe) & (vibe_df["score"].abs() > 0.0)
    ].head(10).to_dict(orient="records")  # Match the number in show_examples
    
    # Extract the example number from the dropdown selection
    try:
        # Parse the example number from the string (e.g., "Example 3: ...")
        example_num = int(example_idx.split(":")[0].replace("Example ", "")) - 1
        if example_num < 0 or example_num >= len(subset):
            return "Example index out of range."
    except (ValueError, AttributeError, IndexError):
        return f"Invalid example selection: '{example_idx}'"
    
    # Get the selected example
    row = subset[example_num]
    print(row)
    print("+++++++++++")
    print("+++++++++++")
    print("+++++++++++")
    print("+++++++++++")
    print("+++++++++++")
    # md = f"**Example for: {selected_vibe.replace('**', '')}**\n\n"
    md = f"**Prompt:** {row['question']}\n\n"
    
    # Show both model outputs if available
    for model in models:
        print(model)
        print(row.keys())
        print("________________________")
        if model in row:
            md += f"**{model} Output:**\n{row[model]}\n________________________\n"
    md += f"**Score:** {row['score']:.3f} "
    
    # Add interpretation of which model exhibits the vibe more
    if len(models) >= 2:
        if row["score"] > 0:
            md += f"({models[0]} exhibits this vibe more)\n\n"
        else:
            md += f"({models[1]} exhibits this vibe more)\n\n"
    
    return md

def update_vibe_dropdown(results_dict):
    """Update the vibe dropdown with choices from results"""
    if not results_dict or "vibe_df" not in results_dict:
        return gr.Dropdown(choices=[], interactive=True)
    
    choices = sorted(results_dict["vibe_df"]["vibe"].unique().tolist())
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None, interactive=True)

def create_vibecheck_ui():
    """
    Builds and returns a Gradio Blocks interface for VibeCheck
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

    with gr.Blocks(theme='davehornik/Tealy', css=custom_css) as demo:
        gr.HTML("""
        <div class="header-text">
            <h1><span class="header-emoji">✨</span> VibeCheck <span class="header-emoji">✨</span></h1>
            <p>Discover and analyze the vibes in your model outputs</p>
        </div>
        """)

        ################################################################
        # LOAD PREVIOUS RESULTS TAB
        ################################################################
        with gr.Tab("Load Previous Results"):
            gr.HTML("""
            <div class="tab-content">
                <h3>Load Previous Analysis Results</h3>
                <p>Select a previously saved analysis to view its results without having to rerun the pipeline.</p>
            </div>
            """)
            
            with gr.Row():
                # Left Column - Selection
                with gr.Column(scale=1):
                    # Dropdown to select saved results
                    saved_results_dropdown = gr.Dropdown(
                        label="Select a saved analysis",
                        choices=list_saved_results(),
                        interactive=True,
                        allow_custom_value=False,
                    )
                    
                    # Refresh button
                    refresh_btn = gr.Button("Refresh List", variant="secondary")
                    
                    # Load button
                    load_btn = gr.Button("Load Selected Results", variant="primary")
                    
                    # Status message
                    load_status = gr.Markdown()
                
            # Output area
            with gr.Row():
                load_output_md = gr.Markdown()
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Vibe Score Distribution")
                    load_output_plot1 = gr.Plot()
                
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model Vibe Scores")
                    load_output_plot2 = gr.Plot()

            # Bottom Row - Vibe Examples
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Explore Vibe Examples")
                    load_vibe_dropdown = gr.Dropdown(
                        label="Select a vibe to see examples",
                        choices=[],
                        interactive=True,
                        multiselect=False,
                        value=None,
                        allow_custom_value=False,
                    )
                    load_example_status = gr.Markdown()  # Status message about number of examples
                    load_example_dropdown = gr.Dropdown(
                        label="Select an example to view",
                        choices=[],
                        interactive=True,
                        multiselect=False,
                        value=None,
                        allow_custom_value=False,
                    )
                    load_examples_md = gr.Markdown()
            
            # Store loaded results state
            load_results_state = gr.State()
            
            # Function to refresh the list of saved results
            def refresh_saved_results():
                return gr.Dropdown(choices=list_saved_results())
            
            refresh_btn.click(
                fn=refresh_saved_results,
                inputs=[],
                outputs=[saved_results_dropdown]
            )
            
            # Function to load selected results
            def on_load_results(selected_result):
                if not selected_result:
                    return "Please select a saved analysis to load.", None, None, None, None
                
                # Get the actual filename from the display name
                filename = selected_result[1] if isinstance(selected_result, tuple) else selected_result
                
                # Load the results
                results = load_saved_result(filename)
                
                if not results or "vibe_df" not in results:
                    return "Failed to load results or invalid result file.", None, None, None, None
                
                # Get model names from the results
                model_names = results.get("models", ["Model A", "Model B"])
                
                # Get accuracy metrics if available
                accuracy_info = ""
                if "vibe_prediction_metrics" in results and results["vibe_prediction_metrics"]:
                    metrics = results["vibe_prediction_metrics"]
                    
                    # Format identity metrics
                    if "identity_metrics" in metrics:
                        id_metrics = metrics["identity_metrics"]
                        accuracy_info += f"\n\n### Model Identity Prediction\n"
                        accuracy_info += f"- Accuracy: {id_metrics.get('accuracy', 'N/A'):.3f} ± {id_metrics.get('acc_std', 'N/A'):.3f}\n"
                        if "acc_ci" in id_metrics:
                            accuracy_info += f"- 95% CI: [{id_metrics['acc_ci'][0]:.3f}, {id_metrics['acc_ci'][1]:.3f}]\n"
                    
                    # Format preference metrics
                    if "preference_metrics" in metrics:
                        pref_metrics = metrics["preference_metrics"]
                        accuracy_info += f"\n### Preference Prediction\n"
                        accuracy_info += f"- Accuracy: {pref_metrics.get('accuracy', 'N/A'):.3f} ± {pref_metrics.get('acc_std', 'N/A'):.3f}\n"
                        if "acc_ci" in pref_metrics:
                            accuracy_info += f"- 95% CI: [{pref_metrics['acc_ci'][0]:.3f}, {pref_metrics['acc_ci'][1]:.3f}]\n"
                
                # Add model names to the summary
                model_info = f"\n\n**Models Compared: {' vs. '.join(model_names)}**\n\n"
                
                # Summarize
                summary_text = (
                    "## Loaded VibeCheck Results\n"
                    # + f"Output directory: {results.get('output_dir', 'N/A')}\n\n"
                    + f"Wandb Run URL: {results.get('wandb_run_url', 'N/A')}\n\n"
                    + model_info
                    + f"**Found {len(results['vibe_df']['vibe'].unique())} vibe(s)**\n"
                    + "\n".join(
                        [
                            f"- {vibe}"
                            for vibe in results["vibe_df"]["vibe"].unique().tolist()
                        ]
                    )
                    + accuracy_info
                )
                
                return (
                    "Results loaded successfully!",
                    summary_text,
                    results["model_vibe_scores_plot"],
                    results["score_dist_plot"],
                    results,
                    gr.Dropdown(choices=sorted(results["vibe_df"]["vibe"].unique().tolist()), 
                               value=sorted(results["vibe_df"]["vibe"].unique().tolist())[0] if results["vibe_df"]["vibe"].unique().tolist() else None)
                )
            
            load_btn.click(
                fn=on_load_results,
                inputs=[saved_results_dropdown],
                outputs=[
                    load_status,
                    load_output_md,
                    load_output_plot1,
                    load_output_plot2,
                    load_results_state,
                    load_vibe_dropdown,
                ],
            )
            
            # Show examples for selected vibe
            load_vibe_dropdown.change(
                fn=show_examples,
                inputs=[load_vibe_dropdown, load_results_state],
                outputs=[load_example_status, load_example_dropdown],
            )

            # Add handler for example dropdown
            load_example_dropdown.change(
                fn=display_selected_example,
                inputs=[load_example_dropdown, load_vibe_dropdown, load_results_state],
                outputs=[load_examples_md],
            )

        ################################################################
        # COMPARE MODELS TAB
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
                    csv_file = gr.File(
                        label="Upload CSV for Model Comparison", 
                        file_types=[".csv"],
                        elem_id="csv-upload"
                    )
                    gr.Examples(
                        examples=["data/friendly_and_cold_sample.csv", "data/cnndm_with_pref.csv"],
                        inputs=csv_file,
                        label="Example Datasets",
                    )
                    csv_columns = gr.Markdown(label="CSV Columns")
                    models = gr.Textbox(
                        label="Model Column Names (comma-separated with a space between each model name)",
                        placeholder="modelA, modelB",
                        elem_id="model-names"
                    )
                    test_mode = gr.Checkbox(
                        label="Test Mode (sample 100 rows for faster results)", 
                        value=False
                    )

                with gr.Column(scale=1):
                    # Add accordion for advanced parameters
                    with gr.Accordion("Advanced Parameters", open=True):
                        # Load the base config to get default values
                        base_config = OmegaConf.load("configs/base.yaml")
                        config_inputs = {}
                        
                        # General Settings
                        with gr.Accordion("General Settings", open=False):
                            config_inputs["project_name"] = gr.Textbox(
                                label="Project Name",
                                value=base_config.project_name,
                                info="Name of the Weights & Biases project where results will be logged."
                            )

                            config_inputs["name"] = gr.Textbox(
                                label="Experiment Name",
                                value=base_config.name,
                                info="Name of the experiment, used for saving the pkl file."
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
                            
                            config_inputs["preference_judge_llm"] = gr.Textbox(
                                label="Preference Judge LLM",
                                value=base_config.preference_judge_llm,
                                info="LLM to use for preference judgment."
                            )
                            
                            config_inputs["num_vibes"] = gr.Number(
                                label="Number of Vibes per Iteration",
                                value=base_config.num_vibes,
                                minimum=1,
                                step=1,
                                info="Maximum number of vibes to use per iteration."
                            )

                            config_inputs["num_final_vibes"] = gr.Number(
                                label="Number of Final Vibes",
                                value=base_config.num_final_vibes,
                                minimum=0,
                                step=1,
                                info="Maximum number of vibes to use in final analysis (if 0, uses all vibes)."
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
                        
                    run_btn = gr.Button("✨ Run VibeCheck ✨", variant="primary", size="lg")

            with gr.Row():
                output_md = gr.Markdown()

            # Plots
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Vibe Score Distribution")
                    output_plot1 = gr.Plot()
                
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model Vibe Scores")
                    output_plot2 = gr.Plot()

            # Vibe Examples
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
                    example_status = gr.Markdown()  # Status message about number of examples
                    example_dropdown = gr.Dropdown(
                        label="Select an example to view",
                        choices=[],
                        interactive=True,
                        multiselect=False,
                        value=None,
                        allow_custom_value=False,
                    )
                    examples_md = gr.Markdown()

            # Store results state
            results_state = gr.State()

            def on_run(
                csv_file,
                model_list_str,
                test_mode,
                *config_args  # Add this to capture all additional config arguments
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
                config_keys = list(config_inputs.keys())  # Get list of config keys
                for i, value in enumerate(config_args):
                    if i < len(config_keys) and value is not None:  # Only include non-None values
                        config_overrides[config_keys[i]] = value

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

                # Get accuracy metrics if available
                accuracy_info = ""
                if "vibe_prediction_metrics" in results and results["vibe_prediction_metrics"]:
                    metrics = results["vibe_prediction_metrics"]
                    
                    # Format identity metrics
                    if "identity_metrics" in metrics:
                        id_metrics = metrics["identity_metrics"]
                        accuracy_info += f"\n\n### Model Identity Prediction\n"
                        accuracy_info += f"- Accuracy: {id_metrics.get('accuracy', 'N/A'):.3f} ± {id_metrics.get('acc_std', 'N/A'):.3f}\n"
                        if "acc_ci" in id_metrics:
                            accuracy_info += f"- 95% CI: [{id_metrics['acc_ci'][0]:.3f}, {id_metrics['acc_ci'][1]:.3f}]\n"
                    
                    # Format preference metrics
                    if "preference_metrics" in metrics:
                        pref_metrics = metrics["preference_metrics"]
                        accuracy_info += f"\n### Preference Prediction\n"
                        accuracy_info += f"- Accuracy: {pref_metrics.get('accuracy', 'N/A'):.3f} ± {pref_metrics.get('acc_std', 'N/A'):.3f}\n"
                        if "acc_ci" in pref_metrics:
                            accuracy_info += f"- 95% CI: [{pref_metrics['acc_ci'][0]:.3f}, {pref_metrics['acc_ci'][1]:.3f}]\n"

                # Summarize
                summary_text = (
                    "## VibeCheck Results\n"
                    + f"Output directory: {results['output_dir']}\n\n"
                    + f"Wandb Run URL: {results['wandb_run_url']}\n\n"
                    + f"### Found {len(results['vibe_df']['vibe'].unique())} vibe(s)"
                    + "\n".join(
                        [
                            f"- {vibe}"
                            for vibe in results["vibe_df"]["vibe"].unique().tolist()
                        ]
                    )
                    + accuracy_info
                )

                return (
                    summary_text,
                    results["model_vibe_scores_plot"],
                    results["score_dist_plot"],
                    results,  # store the dictionary
                )

            # Collect all config inputs for the click handler
            config_input_list = list(config_inputs.values())
            
            run_btn.click(
                fn=on_run,
                inputs=[
                    csv_file,
                    models,
                    test_mode,
                    *config_input_list
                ],
                outputs=[
                    output_md,
                    output_plot1,
                    output_plot2,
                    results_state,
                ],
            )

            # Update vibe dropdown when results change
            results_state.change(
                fn=update_vibe_dropdown,
                inputs=[results_state],
                outputs=[vibe_dropdown],
            )

            # Show examples for selected vibe
            vibe_dropdown.change(
                fn=show_examples,
                inputs=[vibe_dropdown, results_state, models],
                outputs=[example_status, example_dropdown],
            )

            # Add handler for example dropdown
            example_dropdown.change(
                fn=display_selected_example,
                inputs=[example_dropdown, vibe_dropdown, results_state, models],
                outputs=[examples_md],
            )

            # Add the file upload and example handlers
            csv_file.upload(
                fn=on_file_upload, inputs=[csv_file], outputs=[csv_columns]
            )
            csv_file.change(  # Add handler for example clicks
                fn=on_file_upload, inputs=[csv_file], outputs=[csv_columns]
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

def run_webapp():
    app = create_vibecheck_ui()
    app.launch(share=True)

if __name__ == "__main__":
    app = create_vibecheck_ui()
    # To share publicly on a unique link, set share=True
    app.launch(share=True)