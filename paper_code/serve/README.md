# VisDiff API Servers

**make sure you have created a global_vars.py file in the serve directory with the correct API keys. A dummy file is provided in serve/global_vars_dummy.py**

## Design Choices

1. All LLMs/VLMs serve as API with cache enabled, because loading a LLM/VLM is expensive and we never modify them.
2. LLM functions in `utils_llm.py`, VLM functions in `utils_vlm.py`, and others in `utils_general.py`.
3. Write unit tests to understand major functions.

## LLM Server Configuration

1. Set up OpenAI API key: `export OPENAI_API_KEY='[your key]'`
2. Pip install environments: `pip install vllm`
3. Configure global variables in `global_vars.py`
4. Run `python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.5` or any other model hosted on vLLM.
5. Run `python -m serve.utils_llm` to test the LLM.

## VLM Server Configuration (DON'T USE)

1. Install environments:
  - BLIP: `pip install salesforce-lavis`
  - LLaVA: `git clone git@github.com:haotian-liu/LLaVA.git; cd LLaVA; pip install -e .`
2. Configure global variables in `global_vars.py`
3. Run `python serve/vlm_server_[vlm].py`. It takes a while to load the VLM, especially the first time to download the VLM. (Note: concurrency is disabled as it surprisingly leads to worse GPU utilization)
4. Run `python -m serve.utils_vlm` to test the VLM.

