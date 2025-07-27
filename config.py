import os
from dotenv import load_dotenv
load_dotenv()  # reads APIs

MODELS = {
    "GPT-4o": {
        "type": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY"
    },
    "Gemini": {
        "type": "google",
        "model": "gemini-pro",
        "api_key_env": "GOOGLE_API_KEY"
    },
    "Claude": {
        "type": "anthropic",
        "model": "claude-2",
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "Grok": {
        "type": "xai",
        "model": "grok-4",
        "api_key_env": "GROK_API_KEY"
    },
    "Deepseek": {
        "type": "deepseek",
        "model": "deepseek-v3",
        "api_key_env": "DEEPSEEK_API_KEY"
    },
    "llama": {
        "type": "hf",
        "model": "meta-llama/Llama-3.3-70B-Instruct"
    },
    "BioBERT": {
        "type": "hf",
        "model": "dmis-lab/biobert-base-cased-v1.1"
    }
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


