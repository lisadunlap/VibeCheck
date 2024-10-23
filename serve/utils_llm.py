import json
import logging
import os
import threading
from typing import List

import lmdb
import openai
from openai import OpenAI
import anthropic
import datetime
from wandb.sdk.data_types.trace_tree import Trace
import concurrent.futures

from serve.global_vars import LLM_CACHE_FILE, VICUNA_URL, LLM_EMBED_CACHE_FILE, OPENAI_API_KEY, ANTHROPIC_API_KEY, LLAMA_URL, LLAMA3_70B_URL, LLAMA_API_KEY
from serve.utils_general import get_from_cache, save_to_cache, save_emb_to_cache, get_emb_from_cache

logging.basicConfig(level=logging.ERROR)

if not os.path.exists(LLM_CACHE_FILE):
    os.makedirs(LLM_CACHE_FILE)

llm_cache = lmdb.open(LLM_CACHE_FILE, map_size=int(1e11))
llm_embed_cache = lmdb.open(LLM_EMBED_CACHE_FILE, map_size=int(1e11))

openai.api_key = OPENAI_API_KEY
anthropic.api_key = ANTHROPIC_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def get_llm_output(prompt: str, model: str, cache = True, system_prompt = None, history=[], max_tokens=256) -> str:
    openai.api_base = "https://api.openai.com/v1" if model != "llama-3-8b" else LLAMA_URL
    if 'gpt' in model:
        client = OpenAI()
    elif model == "llama-3-8b":
        client = OpenAI(
            base_url=LLAMA_URL,
        )
    elif model == "llama-3-70b":
        openai.api_key = LLAMA_API_KEY
        openai.api_base = LLAMA3_70B_URL
        client = OpenAI(base_url=LLAMA3_70B_URL)
    else:
        client = anthropic.Anthropic()
        
    systems_prompt = "You are a helpful assistant." if not system_prompt else system_prompt

    if "gpt" in model:
        messages = [{"role": "system", "content": systems_prompt}] + history + [
            {"role": "user", "content": prompt},
        ]
    elif 'claude' in model:
        messages = history + [
            {"role": "user", "content": prompt},
        ]
    else:
        # messages = prompt
        messages = [{"role": "system", "content": systems_prompt}] + history + [
            {"role": "user", "content": prompt},
        ]
    key = json.dumps([model, messages])

    cached_value = get_from_cache(key, llm_cache) if cache else None
    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value
    else:
        logging.debug(f"LLM Cache Miss")

    for _ in range(3):
        try:
            if 'gpt-3.5' in model:
                start_time_ms = datetime.datetime.now().timestamp() * 1000
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                end_time_ms = round(datetime.datetime.now().timestamp() * 1000)  # logged in milliseconds
                response = completion.choices[0].message.content.strip()
            elif 'gpt-4' in model:
                start_time_ms = datetime.datetime.now().timestamp() * 1000
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                end_time_ms = round(datetime.datetime.now().timestamp() * 1000)  # logged in milliseconds
                response = completion.choices[0].message.content.strip()
            elif 'claude-opus' in model:
                completion = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    system=systems_prompt,
                )
                response = completion.content[0].text
            elif 'claude' in model:
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
                    extra_body={"stop_token_ids":[128009]}
                )
                response = completion.choices[0].message.content.strip().replace("<|eot_id|>", "")
            elif model == "llama-3-70b":
                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-70B-Instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    extra_body={"stop_token_ids":[128009]}
                )
                response = completion.choices[0].message.content.strip().replace("<|eot_id|>", "")
                
            save_to_cache(key, response, llm_cache)
            return response

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            # if error is Error Code: 400, then it is likely that the prompt is too long, so truncate it
            if "Error code: 400" in str(e):
                messages = [{"role": "system", "content": systems_prompt}] + history + [
                    {"role": "user", "content": prompt[:int(len(prompt) / 2)]},
                ]
            else:
                raise e
    return "LLM Error: Cannot get response."

def get_llm_embedding(prompt: str, model: str) -> str:
    openai.api_base = "https://api.openai.com/v1" if model != "vicuna" else VICUNA_URL
    client = OpenAI()
    key = json.dumps([model, prompt])

    cached_value = get_emb_from_cache(key, llm_embed_cache)

    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value
    else:
        logging.debug(f"LLM Cache Miss")

    for _ in range(3):
        try:
            text = prompt.replace("\n", " ")
            embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
            save_emb_to_cache(key, embedding, llm_embed_cache)
            return embedding
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue

    return "LLM Error: Cannot get response."

def test_get_llm_output():
    prompt = "hello"
    model = "gpt-4"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-3.5-turbo"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "vicuna"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")


if __name__ == "__main__":
    test_get_llm_output()