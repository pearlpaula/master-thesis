from dotenv import load_dotenv
load_dotenv()

import os
import openai
from openai import OpenAI as OpenAIClient
import anthropic
import google.generativeai as genai
from transformers import pipeline
from config import MODELS

def call_llm(name: str, prompt: str, temp: float) -> str:
    cfg = MODELS[name]
    kind = cfg["type"]

    if kind == "openai":
        openai.api_key = os.getenv(cfg["api_key_env"])
        resp = openai.chat.completions.create(
            model=cfg["model"],
            messages=[{"role":"user", "content":prompt}],
            temperature=temp,
            max_tokens=128
        )
        return resp.choices[0].message["content"].strip()

    elif kind == "deepseek":
        client = OpenAIClient(
            api_key=os.getenv(cfg["api_key_env"]),
            base_url="https://api.deepseek.com"
        )
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role":"user","content":prompt}],
            temperature=temp,
            max_tokens=128
        )
        return resp.choices[0].message["content"].strip()

    elif kind == "anthropic":
        client = anthropic.Client(api_key=os.getenv(cfg["api_key_env"]))
        full_prompt = anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT
        resp = client.completions.create(
            model=cfg["model"],
            prompt=full_prompt,
            temperature=temp,
            max_tokens_to_sample=128
        )
        return resp.completion.strip()

    elif kind == "google":
        genai.configure(api_key=os.getenv(cfg["api_key_env"]))
        resp = genai.generate_text(
            model=cfg["model"],
            prompt=prompt,
            temperature=temp,
            max_output_tokens=128
        )
        return resp.candidates[0].text.strip()

    elif kind == "xai":
        client = OpenAIClient(
            api_key=os.getenv(cfg["api_key_env"]),
            base_url="https://api.x.ai/v1"
        )
        resp = client.chat.completions.create(
            model=cfg["model"],     # "grok-4"
            messages=[{"role":"user","content":prompt}],
            temperature=temp,
            max_tokens=128
        )
        return resp.choices[0].message["content"].strip()

    elif kind == "hf":
        model_name = cfg["model"]
        if any(x in model_name.lower() for x in ["llama", "mistral"]):
            gen = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device_map="auto",
                torch_dtype="auto"
            )
            return gen(
                [{"role":"user","content":prompt}],
                max_new_tokens=128,
                do_sample=True,
                temperature=temp
            )[0]["generated_text"]

        # Mask‑fill style (BioBERT)
        elif "biobert" in model_name.lower():
            fill = pipeline("fill-mask", model=model_name, tokenizer=model_name)
            # you need to append a “[MASK]” token to your prompt
            masked = prompt if "[MASK]" in prompt else prompt + " [MASK]"
            candidates = fill(masked, top_k=1)
            return candidates[0]["sequence"]

        # fallback for any other HF text‑gen
        else:
            gen = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device_map="auto",
                torch_dtype="auto"
            )
            return gen(
                prompt,
                max_new_tokens=128,
                do_sample=True,
                temperature=temp
            )[0]["generated_text"]

    else:
        raise ValueError(f"Unknown model type: {kind}")

