# Give your generative models a vibe check :D
### VibeCheck: Discover and Quantify Qualitative Differences in Large Language Models
Lisa Dunlap, Krishna Mandal, Trevor Darrell, Jacob Steinhardt, Joseph E. Gonzalez

Paper link [here](https://arxiv.org/abs/2410.12851), joke version of paper coming soon 

<p align="center">
  <img src="method_vibecheck.png" width="800">
</p>


**Still cleaning this up:** I got distracted trying to implement some causal inference stuff...

## Data

* [Link to chatbot arena data](https://huggingface.co/datasets/lmarena-ai/Llama-3-70b-battles)
* [Human VS GPT (HC3)](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
* [HELM Predictions](https://crfm.stanford.edu/helm/classic/latest/) (fair warning, this is a real pain to download)

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

2. Create a weights and biases account if you dont already have one

3. Copy this into a file named serve/global_vars.py and set your openai key 

```
# LLM API (if you want to use a local LLM, use vLLM)
LLAMA_URL = "http://localhost:8001/v1" 
VICUNA_URL = "http://localhost:8001" 
LLM_CACHE_FILE = "cache/cache_llm"
LLM_EMBED_CACHE_FILE = "cache/cache_llm_embed"

OPENAI_API_KEY = [put your key here]
ANTHROPIC_API_KEY = [put your key here]
```

4. Run a config
```
python main.py --config configs/base.yaml wandb=True
```
This runs a toy example on LLM outputs, one model is prompted to be friendly, the other cold and factual. I randomly assigned preference so friendly results are favored 80% of the time

## Data Structure

All data needs to contain the columns "question", model_name_1, model_name_2, and optionally "preference". If the preference column is not provided, running main will compute the preference via LLM as a jude (warning the LLMs are hardcoded in the file)

Say your two models are gpt-4o and gemini-1.5-flash. Your CSV should have the columns "question", "gpt-4o", "gemini-1.5-flash" and in your config, set your data path and set `models: [gpt-4o, gemini-1.5-flash]`. Sometime soon I will add an option to only optimize for model matching if you only care to find differentiating qualities, so get excited for that. 

## Code Structure (more explanation coming soon)

This code structure is loosely modeled off the [VisDiff repo](https://github.com/Understanding-Visual-Datasets/VisDiff)

Here are the core components:
* [Proposer](components/proposer.py): takes in prompt, output_a, output_b triplets and return a list of axes
* [Reducer](components/reducer.py): takes a long list of axes and returns a shorter list of representative axes
* [Ranker](components/ranker.py): takes in a triplet and an axis and produces a score

## 🎯 Citation

If you use this repo in your research, please cite it as follows and ideally use the word 'vibe' in said research:
```
@article{dunlap_vibecheck,
  title={VibeCheck: Discover and Quantify Qualitative Differences in Large Language Models},
  author={Lisa Dunlap and Krishna Mandal and Trevor Darrell and Jacob Steinhardt and Joseph E Gonzalez},
  journal={arXiv preprint arXiv:2312.02974},
  year={2024},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2410.12851},
}
```
