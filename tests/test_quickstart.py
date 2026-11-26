"""Offline smoke test of the README quickstart, with the LLM calls stubbed out.

Run with `pytest tests/` or `python tests/test_quickstart.py` from a fresh
`pip install -r requirements.txt` environment. No API keys are needed.
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

VIBES = [
    "Friendliness: High: warm and enthusiastic, uses exclamation marks. Low: cold and matter-of-fact.",
    "Length: High: long and detailed. Low: short and terse.",
]


def fake_llm(prompt, model, cache=True, system_prompt=None, history=[], max_tokens=256):
    if isinstance(prompt, list):
        return [fake_llm(p, model) for p in prompt]
    if "Response A:" in prompt:  # ranker: pick the friendlier / longer response
        a = prompt.split("Response A:")[1].split("Response B:")[0]
        b = prompt.split("Response B:")[1].split("Properties (restated)")[0]
        lines = ["Ranking:"]
        for i, vibe in enumerate(VIBES, 1):
            fa, fb = (a.count("!"), b.count("!")) if vibe.startswith("Friendliness") else (len(a), len(b))
            lines += [f"Property {i}: {'A' if fa > fb else 'B' if fb > fa else 'equal'}", "Analysis: stub"]
        return "\n".join(lines)
    return "\n".join("- " + v for v in VIBES)  # proposer, reducer and question types


def test_quickstart_example():
    os.environ["WANDB_MODE"] = "offline"
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())  # utils_llm creates its cache/ dir in the working directory
    try:
        _run_quickstart()
    finally:
        os.chdir(cwd)


def _run_quickstart():
    from omegaconf import OmegaConf
    import main
    import components.propose
    import components.rank  # imported lazily by main, after the proposer's LLM calls

    for module in (main, components.propose, components.rank):
        module.get_llm_output = fake_llm

    config = OmegaConf.merge(
        OmegaConf.load(os.path.join(REPO, "configs/base.yaml")),
        OmegaConf.create({
            "data_path": os.path.join(REPO, "data/friendly_and_cold_sample.csv"),
            "models": ["friendly", "cold"],
            "num_final_vibes": 3,
            "wandb": False,
        }),
    )
    results = main.main(config)
    scores = results["scores_df"].set_index("vibe")["score"]
    assert scores[VIBES[0]] > 0.5  # the friendly model is the friendlier one


if __name__ == "__main__":
    test_quickstart_example()
    print("ok")
