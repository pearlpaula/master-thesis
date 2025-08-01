import os
from dotenv import load_dotenv
load_dotenv()  # read APIs

MODELS = {
    # lightweight models for main run
    "GPT-3.5":   {"type": "openai",    "model": "gpt-3.5-turbo"},
    "Gemini":    {"type": "google",    "model": "gemini-2.5-flash"},
    "Claude":    {"type": "anthropic", "model": "claude-3-5-haiku-20241022"},
    "Grok":      {"type": "xai",       "model": "grok-3-mini"},
    "Deepseek":  {"type": "deepseek",  "model": "deepseek-v2.5-1210"},
    
    # open-weight baselines
    "llama":     {"type": "hf",        "model": "meta-llama/Meta-Llama-3-70B-Instruct"},  
    "mistral":   {"type": "hf",        "model": "mistralai/Mistral-7B-Instruct-v0.3"},
    "biobert":   {"type": "hf",        "model": "dmis-lab/biobert-base-cased-v1.2"},

    # heavy models for validation
    "GPT-4o":        {"type": "openai",    "model": "gpt-4o"},
    "Gemini-Pro":    {"type": "google",    "model": "gemini-2.5-pro"},
    "Claude-Sonnet": {"type": "anthropic", "model": "claude-3-sonnet"},
    "Grok-4":        {"type": "xai",       "model": "grok-4"},
    "Deepseek-R1":   {"type": "deepseek",  "model": "deepseek-reasoner"}
}


TEMPERATURES = [0.2, 0.7, 1.0]

PROMPT_STYLES = {
    "conversational": "Q: {context} {question}\nA:",
    "generative":    "Context: {context}\nWrite a detailed answer to: {question}"
}

# BBQ github
BBQ_BASE_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/"
BBQ_FILES = [
    "Age.jsonl", "Disability_status.jsonl", "Gender_identity.jsonl",
    "Nationality.jsonl", "Physical_appearance.jsonl", "Race_ethnicity.jsonl",
    "Race_x_SES.jsonl", "Race_x_gender.jsonl", "Religion.jsonl",
    "SES.jsonl", "Sexual_orientation.jsonl"
]


