from dotenv import load_dotenv
load_dotenv()

import sys
import pandas as pd
import random
from tqdm.auto import tqdm

from experiments.call_llm import call_llm
from config import MODELS, TEMPERATURES, PROMPT_STYLES
from load import load_bbq

# open‑source models (full corpus)
FREE_MODELS = {"llama", "mistral", "biobert"}

def stratified_sample(df, n_per_cat=100, seed=42):
    """Return n_per_cat examples per 'category' from df."""
    return (
        df.groupby("category", group_keys=False)
          .apply(lambda g: g.sample(n=min(n_per_cat, len(g)), random_state=seed))
          .reset_index(drop=True)
    )

def run_for_model(model_name: str):
    df_full = load_bbq()
    records = []

    is_free = model_name in FREE_MODELS

    # decide which rows to run
    if is_free:
        to_run = df_full.itertuples()
    else:
        paid_slice = stratified_sample(df_full, n_per_cat=100)
        to_run = paid_slice.itertuples()

    desc = f"Running {model_name}"
    for row in tqdm(to_run, total=len(df_full) if is_free else len(paid_slice), desc=desc):
        for style, template in PROMPT_STYLES.items():
            prompt = template.format(context=row.context, question=row.question)
            for temp in TEMPERATURES:
                out = call_llm(model_name, prompt, temp)
                rec = row._asdict()
                rec.update({
                    "model": model_name,
                    "prompt_style": style,
                    "temperature": temp,
                    "response": out
                })
                records.append(rec)

    results = pd.DataFrame(records)
    out_path = f"results/llm_responses_{model_name}.parquet"
    results.to_parquet(out_path, index=False)
    print(f"Done! {model_name}: collected {len(results)} rows → {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m experiments.run_experiments <model_name>")
        print("Available models:", ", ".join(MODELS.keys()))
        sys.exit(1)

    model = sys.argv[1]
    if model not in MODELS:
        print(f"Error: unknown model '{model}'.  Choose from {list(MODELS.keys())}")
        sys.exit(1)

    run_for_model(model)


