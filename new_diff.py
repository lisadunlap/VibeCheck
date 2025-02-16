import openai
from datasets import load_dataset
from PIL import Image
import io
import wandb
import os
import pandas as pd
import base64

# Load dataset
df = pd.read_csv("open-binary-converted.csv")

# OpenAI API setup
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Use environment variable for security
openai.api_key = OPENAI_API_KEY

# Initialize wandb
wandb.init(project="VibeCheck", name="image_judging")

# Function to convert binary data into an image
def get_image_from_binary(image_data):
    image_bytes = eval(image_data)["bytes"]  # Convert string to dictionary and get bytes
    return Image.open(io.BytesIO(image_bytes))

# Function to encode image as base64 for OpenAI API
def encode_image(image):
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")
    
# The following are prompts.    
    


def query_gpt_4o(image1, image2, prompt, preference):
    """Send image comparison prompt to GPT-4o"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a machine learning researcher analyzing image model differences."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

# Iterate over dataset and process images
for row in df.iterrows():
    proposer_prompt_freeform = """
You are a machine learning researcher trying to figure out the major differences between the behaviors of two image generation models by finding differences in their responses to the same set of questions and seeing if these differences correspond with user preferences. Write down as many differences as you can find between the two outputs. Please format your differences as a list of properties that appear more in one output than the other.

Below are multiple sets of questions and responses, separated by dashed lines. For each set, analyze the differences between Model 1 and Model 2. What properties are seen in the responses from Model 1 that are not seen in the responses from Model 2? What properties are seen in the responses from Model 2 that are not seen in the responses from Model 1?

{combined_responses}

The format should be a list of properties that appear more in one output than the other in the format of a short description of the property. An example of a possible output is,
- "vibrant neon color scheme"
- "oil painting texture with visible brushstrokes"
- "cinematic lighting with dramatic shadows"
- "minimalist flat design composition"
- "surreal dreamlike elements"
- "analog film grain effect"

Note that this example is not at all exhaustive, but rather just an example of the format. Consider differences on many different axes such as art style, color palette and contrast, composition and perspective, texture and detail level, lighting and shadow, and any other axis that you can think of. 
    
Remember that these properties should be human interpretable and that the differences should be concise (<= 10 words), substantive and objective. Write down as many properties as you can find. Do not explain which model has which property, simply describe the property.
If there are no substantive differences between the outputs, please respond with only "No differences found."
"""

    judge_systems_prompt = """You are a fair and unbiased judge. Your task is to compare the outputs of two image models (A and B) on a given property. Which image better aligns more with the given property, A, B, or equal?

    Your sole focus is to determine which image better aligns with the given property, NOT how good or bad the image is. Avoid any position bias and remain as objective as possible. Consider what the property means and how it applies to the images. Would a reasonable person be able to tell which image aligns more with the property based on the description?

    Instructions:
        • If Image A aligns with the property more than Image B, respond with "A".
        • If Image B aligns with the property more than Image A, respond with "B".
        • If the images are roughly equal on the property, respond with "equal".
        • If the property does not apply to these images (e.g., the property is about color contrast, but the images are black and white), respond with "N/A".
        • If you are unsure about the meaning of the property, respond with "unsure". Think about if a reasonable person would find the property easy to understand.

    A group of humans should agree with your decision. Use the following format for your response:
    Explanation: {{your explanation}}
    Model: {{A, B, equal, N/A, or unsure}}

    Remember to be as objective as possible and strictly adhere to the response format."""

    ranker_prompt1 = (
            judge_systems_prompt
            + """
    Here is the property and the two responses:
    {ranker_inputs}

    Remember to be as objective as possible and strictly adhere to the response format.
    """
        )

    ranker_prompt2 = (
            judge_systems_prompt
            + """
    Here is the property and the two responses:
    {ranker_inputs_reversed}

    Remember to be as objective as possible and strictly adhere to the response format.
    """
        )

    if row["preference"] == "black-forest":
        preference = 1
    else:
        preference = 2
    black_forest_img, stable_diff_img = get_image_from_binary(row["black_forest"]), get_image_from_binary(row["stable_diffusion"])
    result = query_gpt_4o(encode_image(black_forest_img), encode_image(stable_diff_img), proposer_prompt_freeform, preference)
    print("Differences:", result)

    # Log the results to wandb
    wandb.log({
        "chosen_image": wandb.Image(black_forest_img),
        "rejected_image": wandb.Image(stable_diff_img),
        "differences": result
    })

# Finish wandb run
wandb.finish()