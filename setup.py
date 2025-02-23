from setuptools import setup, find_packages

setup(
    name="vibecheck",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "tqdm",
        "wandb",
        "openai",
        "anthropic",
        "plotly",
        "gradio",
        "torch",
        "scikit-learn"
    ]
)