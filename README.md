# Give your generative models a ✨vibe check✨


### [VibeCheck: Discover and Quantify Qualitative Differences in Large Language Models](https://bench-mark.org)

<p align="center">
  <img src="method_vibecheck.png" width="800">
</p>


**This is a simplified and more user-friendly version of the VibeCheck paper.** Original code is in `paper_code` and should run, it's just very messy. 
<!-- Still working on adding all the functionality of the orignal code but the core functionality is here and the visualizations are much better. Namely we moved to using [LOTUS](https://lotus-ai.readthedocs.io/en/latest/), a pandas wrapper to easily run LLM/embedding calls on your data. It reduced my many thousand lines of code to like 2 files. I'm telling you it's the bees knees. -->

## Data

* [Link to chatbot arena data](https://huggingface.co/datasets/lmarena-ai/Llama-3-70b-battles)
* [Human VS GPT (HC3)](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
* [HELM Predictions](https://crfm.stanford.edu/helm/classic/latest/)

## Quickstart

0. (Recommended) Create a new conda environment.
   
  ```
  conda create -n myenv python=3.10 -y
  conda activate myenv
  ```

1. Installation (*please make a PR if I forgot any imports!*)
```
pip install -r requirements.txt
```

2. Create a [weights and biases account](https://wandb.ai/site) if you dont already have one

3. Set env variables for your LLM API keys (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY, etc)

<!-- To run local models, you can use the [LiteLLM library](https://docs.litellm.ai/docs/) with notes on how to set up with LOTUS [here](https://lotus-ai.readthedocs.io/en/latest/llm.html) -->

4. Example run
```
python main.py data_path=data/friendly_and_cold_sample.csv models=[friendly,cold] num_final_vibes=3
```
This runs a toy example on LLM outputs, one model is prompted to be friendly, the other cold and factual. I randomly assigned preference so friendly results are favored 80% of the time

Alternatively, you can set a custom [config](configs/base.yaml) and run with `python main.py --config configs/my_config.yaml [any other args you want to override]`

## Data Structure

All data needs to contain the columns "question", model_name_1, model_name_2, and optionally "preference". If the preference column is not provided, run `generate_preference_labels.py` to compute the preference via LLM as a judge.

Say your two models are gpt-4o and gemini-1.5-flash. Your CSV should have the columns "question", "gpt-4o", "gemini-1.5-flash" and in your command, set your data path and set `models=['gpt-4o', 'gemini-1.5-flash']`. If you only care to find differentiating qualities, you can set `filter.min_pref_score_diff=0`.

## 🎯 Citation

If you use this repo in your research, please cite it as follows and ideally use the word 'vibe' in said research:
```
@article{dunlap_vibecheck,
  title={VibeCheck: Discover and Quantify Qualitative Differences in Large Language Models},
  author={Lisa Dunlap and Krishna Mandal and Trevor Darrell and Jacob Steinhardt and Joseph E Gonzalez},
  journal={International Conference on Learning Representations},
  year={2025},
  url={https://arxiv.org/abs/2410.12851},
}
```

