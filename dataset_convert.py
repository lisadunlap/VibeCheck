import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the CSV file
input_path = "open-binary.csv"  
output_path = "open-binary-converted.csv"  
df = pd.read_csv(input_path)

image_data = eval(df["chosen"].iloc[0])["bytes"]  # Convert string representation to dictionary
image = Image.open(io.BytesIO(image_data))

allowed_models = {"stabilityai/stable-diffusion-3.5-large", "black-forest-labs/FLUX.1-dev"} 
model_a = "stabilityai/stable-diffusion-3.5-large"
model_b = "black-forest-labs/FLUX.1-dev"

df = df[
    ((df["chosen_model"] == model_a) & (df["rejected_model"] == model_b)) | 
    ((df["chosen_model"] == model_b) & (df["rejected_model"] == model_a))
]

df["black-forest"] = df.apply(
    lambda row: row["chosen"] if row["chosen_model"] == model_b else row["rejected"],
    axis = 1
)

df["stable-diffusion"] = df.apply(
    lambda row: row["chosen"] if row["chosen_model"] == model_a else row["rejected"],
    axis = 1
)

df["preference"] = df.apply(
    lambda row: "black-forest" if row["chosen_model"] == model_b else "stable-diffusion",
    axis = 1
)

df = df[[
    "prompt", "black-forest", "stable-diffusion", "preference"
]]

df = df.rename(columns = {"prompt": "question"})    
#df = df.rename(columns = {"prompt": "question", "model_name_1": "a_type", "model_name_2": "b_type", "response_a": "claude", "response_b": "gpt"})
# # Read the original CSV
# input_path = "data/arena_human_preference_train.csv"  # Replace with input CSV file path
# output_path = "data/output_with_preference.csv"  # Replace with output CSV file path

# # Load the CSV
# df = pd.read_csv(input_path)

# allowed_models = {"claude-2.1", "gpt-4-1106-preview"}
# df = df[df[["model_name_1", "model_name_2"]].apply(set, axis=1) == allowed_models]

# def reorder_models(row):
#     if row["model_name_1"] == "gpt-4-1106-preview":
#         # Swap the models if necessary
#         row["model_name_1"], row["model_name_2"] = row["model_name_2"], row["model_name_1"]
#         row["response_a"], row["response_b"] = row["response_b"], row["response_a"]
#         if not row["winner_tie"]:
#             row["winner_model_a"], row["winner_model_b"] = row["winner_model_b"], row["winner_model_a"]
#     return row

# df = df.apply(reorder_models, axis=1)

# # Map winner columns to the new 'preference' column
# def map_preference(row):
#     if row["winner_model_a"] == 1:
#         return row["model_name_1"]
#     elif row["winner_model_b"] == 1:
#         return row["model_name_2"]
#     elif row["winner_tie"] == 1:
#         return "tie"
#     else:
#         return None  # Handle unexpected cases

# df["preference"] = df.apply(map_preference, axis=1)

# # Drop the original winner columns if they are no longer needed
# df = df.drop(columns=["id", "winner_model_a", "winner_model_b", "winner_tie"])

# print(df.head(10))

# df = df.rename(columns = {"prompt": "question", "model_name_1": "a_type", "model_name_2": "b_type", "response_a": "claude", "response_b": "gpt"})

# # Reorder question to the front
# new_order = ['question', 'claude', 'gpt', 'a_type', 'b_type', 'preference']
# df = df[new_order]

# df.loc[df['a_type'] =='claude-2.1', 'a_type'] = 'claude'
# df.loc[df['b_type'] == 'gpt-4-1106-preview', 'b_type'] = 'gpt'
# df.loc[df['preference'] == 'gpt-4-1106-preview', 'preference'] = 'gpt'
# df.loc[df['preference'] == 'claude-2.1', 'preference'] = 'claude'

# def extract_first_two(question_str):
#     question_str = question_str.strip('[]').replace('"', '')
#     questions = question_str.split(",", 2)
#     return questions[:2]
    
# df['question'] = df['question'].apply(extract_first_two)

print(df.head(5))

# Save the updated CSV
df.to_csv(output_path, index=False)
print(f"Converted CSV saved to {output_path}")
