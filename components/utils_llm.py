import json
import logging
import os
import threading
from typing import List
import concurrent.futures
from tqdm import tqdm

import lmdb
import openai
from openai import OpenAI
import anthropic
import datetime
import numpy as np

from components.utils_general import (
    get_from_cache,
    save_to_cache,
    save_emb_to_cache,
    get_emb_from_cache,
)

logging.basicConfig(level=logging.ERROR)

if not os.path.exists("cache/llm_cache"):
    os.makedirs("cache/llm_cache")

if not os.path.exists("cache/llm_embed_cache"):
    os.makedirs("cache/llm_embed_cache")

llm_cache = lmdb.open("cache/llm_cache", map_size=int(1e11))
llm_embed_cache = lmdb.open("cache/llm_embed_cache", map_size=int(1e11))

cache_lock = threading.Lock()

def get_llm_output(
    prompt: str | List[str], model: str, cache=True, system_prompt=None, history=[], max_tokens=256
) -> str | List[str]:
    if isinstance(prompt, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(
                    get_llm_output, p, model, cache, system_prompt, history, max_tokens
                )
                for p in prompt
            ]
            concurrent.futures.wait(futures)
            return [future.result() for future in futures]

    openai.api_base = (
        "https://api.openai.com/v1" if model != "llama-3-8b" else "http://localhost:8000/v1"
    )
    if "gpt" in model:
        client = OpenAI()
    elif model == "llama-3-8b":
        client = OpenAI(
            base_url="http://localhost:8000/v1",
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
        messages = (
            [{"role": "system", "content": systems_prompt}]
            + history
            + [
                {"role": "user", "content": prompt},
            ]
        )
    key = json.dumps([model, messages])

    with cache_lock:
        cached_value = get_from_cache(key, llm_cache) if cache else None

    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value
    else:
        logging.debug(f"LLM Cache Miss")

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
                )
                response = completion.choices[0].message.content.strip()
            elif "gpt-4" in model:
                start_time_ms = datetime.datetime.now().timestamp() * 1000
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                end_time_ms = round(
                    datetime.datetime.now().timestamp() * 1000
                )
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
                    temperature=0.7,
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

            with cache_lock:
                save_to_cache(key, response, llm_cache)

            return response

        except Exception as e:
            logging.error(f"LLM Error: {e}")
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
    return "LLM Error: Cannot get response."


from components.utils_text_embedding import get_text_embedding
def get_llm_embedding(prompt: str | List[str], model: str, instruction: str = "", cache=True) -> str | List[str]:
    if isinstance(prompt, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(get_llm_embedding, p, model, instruction, cache)
                for p in prompt
            ]
            concurrent.futures.wait(futures)
            return [future.result() for future in futures]

    # Truncate text if it's too long (approximately 8192 tokens)
    MAX_CHARS = 32768  # ~8192 token
    if len(str(prompt)) > MAX_CHARS:
        logging.warning(f"Truncating text from {len(prompt)} to {MAX_CHARS} characters")
        prompt = str(prompt)[:MAX_CHARS]

    openai.api_base = "https://api.openai.com/v1"
    client = OpenAI()
    if len(instruction) > 0:
        key = json.dumps([model, prompt, instruction])
    else:
        key = json.dumps([model, prompt])

    with cache_lock:
        cached_value = get_emb_from_cache(key, llm_embed_cache) if cache else None

    if cached_value is not None:
        logging.debug(f"LLM Embedding Cache Hit")
        return cached_value
    else:
        logging.debug(f"LLM Embedding Cache Miss")

    for _ in range(3):
        try:
            if model == "nvidia/NV-Embed-v2":
                print(f"Getting embeddings for {prompt} from embedding server")
                embedding = get_text_embedding([prompt], instruction, server_url="http://localhost:5000")[0]
            else:
                text = prompt.replace("\n", " ")
                embedding = (
                    client.embeddings.create(input=[text], model=model).data[0].embedding
                )
            
            with cache_lock:
                save_emb_to_cache(key, embedding, llm_embed_cache)
                
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue

    return None


def test_get_llm_output():
    prompt = "hello"
    # model = "gpt-4"
    # completion = get_llm_output(prompt, model)
    # print(f"{model=}, {completion=}")
    # model = "gpt-3.5-turbo"
    # completion = get_llm_output(prompt, model)
    # print(f"{model=}, {completion=}")
    model = "llama-3-8b"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")

def test_get_llm_embedding():
    prompt = "hello"
    model = "nvidia/NV-Embed-v2"
    embedding = get_llm_embedding(prompt, model, instruction="", cache=False)
    print(f"{model=}, {np.array(embedding).shape}")
    embedding = embedding / np.linalg.norm(embedding)
    prompt2 = "hello"
    far_prompt = "what is the capital of the moon?"
    embedding2 = get_llm_embedding(prompt2, model, cache=False)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    print(f"{model=}, {np.array(embedding2).shape}")
    print(f"cosine similarity (should be close to 1): {np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))}")
    far_embedding = get_llm_embedding(far_prompt, model, cache=False)
    far_embedding = far_embedding / np.linalg.norm(far_embedding)
    print(f"cosine similarity: {np.dot(embedding, far_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(far_embedding))}")
    model = "text-embedding-3-small"
    embedding = get_llm_embedding(prompt, model, cache=False)
    print(f"{model=}, {np.array(embedding).shape}")

    prompt2 = "hello"
    far_prompt = "what is the capital of the moon?"
    embedding2 = get_llm_embedding(prompt2, model, cache=False)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    print(f"{model=}, {np.array(embedding2).shape}")
    print(f"cosine similarity (should be close to 1): {np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))}")
    far_embedding = get_llm_embedding(far_prompt, model, cache=False)
    far_embedding = far_embedding / np.linalg.norm(far_embedding)
    print(f"cosine similarity: {np.dot(embedding, far_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(far_embedding))}")

if __name__ == "__main__":
    test_get_llm_output()
    test_get_llm_embedding()