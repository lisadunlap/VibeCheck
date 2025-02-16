import json
import logging
import os
import threading
from typing import List
import concurrent.futures
from tqdm import tqdm
import io

import lmdb
import openai
from openai import OpenAI
import anthropic
import datetime
import numpy as np
import base64

from components.utils_general import (
    get_from_cache,
    save_to_cache,
    save_emb_to_cache,
    get_emb_from_cache,
)

logging.basicConfig(level=logging.ERROR)

if not os.path.exists("cache/vlm_cache"):
    os.makedirs("cache/vlm_cache")

if not os.path.exists("cache/vlm_embed_cache"):
    os.makedirs("cache/vlm_embed_cache")

vlm_cache = lmdb.open("cache/vlm_cache", map_size=int(1e11))
vlm_embed_cache = lmdb.open("cache/vlm_embed_cache", map_size=int(1e11))

# Function to convert binary data into an image
def get_image_from_binary(image_data):
    image_bytes = eval(image_data)["bytes"]  # Convert string to dictionary and get bytes
    return Image.open(io.BytesIO(image_bytes))

# Function to encode image as base64 for OpenAI API
def encode_image(image):
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")
    
    
def get_vlm_output(
    prompt: str | List[str], model: str, cache=True, system_prompt=None, history=[], max_tokens=256
) -> str | List[str]:
    # Handle list of prompts with thread pool
    if isinstance(prompt, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(
                    get_vlm_output, p, model, cache, system_prompt, history, max_tokens
                )
                for p in prompt
            ]
            return [f.result() for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures))]

    # Original single prompt logic
    openai.api_base = (
        "https://api.openai.com/v1" if model != "llama-3-8b" else "http://localhost:8001/v1"
    )
    if "gpt" in model:
        client = OpenAI()
    elif model == "llama-3-8b":
        client = OpenAI(
            base_url="http://localhost:8001/v1",
        )
    else:
        client = anthropic.Anthropic()

    systems_prompt = (
        "You are a helpful assistant." if not system_prompt else system_prompt
    )

    if "gpt" in model:
        messages = (
            [{"role": "system", "content": systems_prompt}]
            + history
            + [
                {"role": "user", "content": prompt},
            ]
        )
    elif "claude" in model:
        messages = history + [
            {"role": "user", "content": prompt},
        ]
    else:
        # messages = prompt
        messages = (
            [{"role": "system", "content": systems_prompt}]
            + history
            + [
                {"role": "user", "content": prompt},
            ]
        )
    key = json.dumps([model, messages])

    cached_value = get_from_cache(key, vlm_cache) if cache else None
    if cached_value is not None:
        logging.debug(f"VLM Cache Hit")
        return cached_value
    else:
        logging.debug(f"VLM Cache Miss")

    for _ in range(3):
        try:
            if "gpt-3.5" in model:
                start_time_ms = datetime.datetime.now().timestamp() * 1000
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                end_time_ms = round(
                    datetime.datetime.now().timestamp() * 1000
                )  # logged in milliseconds
                response = completion.choices[0].message.content.strip()
            elif "gpt-4" in model:
                start_time_ms = datetime.datetime.now().timestamp() * 1000
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                end_time_ms = round(
                    datetime.datetime.now().timestamp() * 1000
                )  # logged in milliseconds
                response = completion.choices[0].message.content.strip()
            elif "claude-opus" in model:
                completion = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    system=systems_prompt,
                )
                response = completion.content[0].text
            elif "claude" in model:
                completion = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                response = completion.content[0].text
            elif model == "vicuna":
                completion = client.chat.completions.create(
                    model="lmsys/vicuna-7b-v1.5",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,  # TODO: greedy may not be optimal
                )
                response = completion.choices[0].message.content.strip()
            elif model == "llama-3-8b":
                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    extra_body={"stop_token_ids": [128009]},
                )
                response = (
                    completion.choices[0]
                    .message.content.strip()
                    .replace("<|eot_id|>", "")
                )

            save_to_cache(key, response, vlm_cache)
            return response

        except Exception as e:
            logging.error(f"VLM Error: {e}")
            # if error is Error Code: 400, then it is likely that the prompt is too long, so truncate it
            if "Error code: 400" in str(e):
                messages = (
                    [{"role": "system", "content": systems_prompt}]
                    + history
                    + [
                        {"role": "user", "content": prompt[: int(len(prompt) / 2)]},
                    ]
                )
            else:
                raise e
    return "VLM Error: Cannot get response."


def get_vlm_embedding(prompt: str | List[str], model: str) -> str | List[str]:
    # Handle list of prompts with thread pool
    if isinstance(prompt, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(get_vlm_embedding, p, model)
                for p in prompt
            ]
            return [f.result() for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures))]

    # Original single prompt logic
    openai.api_base = "https://api.openai.com/v1"
    client = OpenAI()
    key = json.dumps([model, prompt])

    cached_value = get_emb_from_cache(key, vlm_embed_cache)

    if cached_value is not None:
        logging.debug(f"VLM Cache Hit")
        return cached_value
    else:
        logging.debug(f"VLM Cache Miss")

    for _ in range(3):
        try:
            text = prompt.replace("\n", " ")
            embedding = (
                client.embeddings.create(input=[text], model=model).data[0].embedding
            )
            save_emb_to_cache(key, embedding, vlm_embed_cache)
            
            return embedding
        except Exception as e:
            logging.error(f"VLM Error: {e}")
            continue

    return None


def test_get_vlm_output():
    prompt = "hello"
    model = "gpt-4"
    completion = get_vlm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-3.5-turbo"
    completion = get_vlm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "vicuna"
    completion = get_vlm_output(prompt, model)
    print(f"{model=}, {completion=}")


if __name__ == "__main__":
    test_get_vlm_output()
