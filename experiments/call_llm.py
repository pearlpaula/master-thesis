from dotenv import load_dotenv
load_dotenv()

import os
from config import MODELS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import anthropic, openai
from google import genai
from xai_sdk import Client
from xai_sdk.chat import user

PIPE_CACHE = {}

# Initialize API clients
from openai import OpenAI
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_xai = Client(api_key=os.getenv("GROK_API_KEY"))
client_google = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def call_llm(name, prompt, temp):
    cfg = MODELS[name]
    t = cfg["type"]

    # ----- OpenAI -----
    if t == "openai":
        resp = client_openai.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temp, max_tokens=128
        )
        return resp.choices[0].message.content.strip()

    # ----- Anthropic Claude -----
    if t == "anthropic":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=cfg["model"], temperature=temp, max_tokens=128,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()

    # ----- Google Gemini -----
    if t == "google":
        resp = client_google.models.generate_content(
            model=cfg["model"],
            contents=prompt,
            config={"temperature": temp, "max_output_tokens": 128}
        )
        return resp.text.strip()

    # ----- DeepSeek -----
    if t == "deepseek":
        client = openai.OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temp, max_tokens=128
        )
        return resp.choices[0].message.content.strip()

    # ----- XAI Grok -----
    if t == "xai":
        chat = client_xai.chat.create(model=cfg["model"], temperature=temp)
        chat.append(user(prompt))
        response = chat.sample()
        return response.content.strip()

    # ----- HuggingFace (LLaMA, Mistral, BioBERT) -----
    if t == "hf":
        if name not in PIPE_CACHE:
            if "biobert" in cfg["model"].lower():
                tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
                model = AutoModelForMaskedLM.from_pretrained(cfg["model"])
                PIPE_CACHE[name] = pipeline("fill-mask", model=model, tokenizer=tokenizer, device_map="auto")
            else:
                tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
                model = AutoModelForCausalLM.from_pretrained(
                    cfg["model"], device_map="auto", torch_dtype="auto"
                )
                PIPE_CACHE[name] = pipeline("text-generation", model=model, tokenizer=tokenizer)

        pipe = PIPE_CACHE[name]
        if "biobert" in cfg["model"].lower():
            masked_prompt = prompt.replace("Answer:", "[MASK]") if "Answer:" in prompt else prompt.strip() + " [MASK]"
            return pipe(masked_prompt, top_k=1)[0]["sequence"]
        else:
            return pipe(prompt, max_new_tokens=128, temperature=temp)[0]["generated_text"]

    raise ValueError(f"Unsupported model type: {t}")

