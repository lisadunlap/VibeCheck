from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Create output directory
output_dir = Path("Open-binarized")
output_dir.mkdir(exist_ok=True)

# Load JUST THE METADATA (no images)
print("Loading metadata...")
ds = load_dataset(
    "data-is-better-together/open-image-preferences-v1-binarized",  # URL-only version
    split="train",  
    trust_remote_code=True
)

# Convert dataset to Pandas DataFrame
df = pd.DataFrame(ds)

# Save metadata to CSV with progress tracking
csv_path = output_dir / "metadata.csv"
print(f"Saving metadata to {csv_path}...")
with tqdm(total=len(df), desc="Saving CSV", unit="rows") as pbar:
    df.to_csv(csv_path, index=False)
    pbar.update(len(df))

print("Dataset download and saving complete!")



# LLM arena data
# import pandas as pd

# df = pd.read_csv("hf://datasets/lmarena-ai/arena-human-preference-55k/train.csv")

# output_path = "arena_human_preference_train.csv"
# df.to_csv(output_path, index=False)

# print(f"File saved to {output_path}")

# # Convert to question, llm_1_response, llm_2_response, preference
# # prompt is question, outside of these four we should discard everything else 