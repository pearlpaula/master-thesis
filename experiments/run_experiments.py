import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from experiments.call_llm import call_llm
from config import MODELS, TEMPERATURES, 
from load import load_bbq  # <--- Import your loading function

ROOT_DIR = "master_thesis"
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

bbq_df = load_bbq()

if "prompt" not in bbq_df.columns:
    bbq_df["prompt"] = bbq_df["context"] + " " + bbq_df["question"]

required_cols = ["example_id", "category", "prompt_style", "prompt"]
for col in required_cols:
    if col not in bbq_df.columns:
        raise ValueError(f"Column {col} is missing from the BBQ dataframe!")



# sample 100 prompts per category
subset = (
    bbq_df.groupby("category", group_keys=False)
    .apply(lambda x: x.sample(n=100, random_state=42))
    .reset_index(drop=True)
)

print(f"Running model: {MODELS} on {len(subset)} prompts × {len(TEMPERATURES)} temperatures")

# create batch
BATCH_SIZE = 4 if MODELS.lower() in ["gemini", "grok", "claude", "gpt-4o", "deepseek"] else 8

records = []

# main loop
for temp in TEMPERATURES:
    print(f"\nProcessing temperature: {temp}")

    for i in tqdm(range(0, len(subset), BATCH_SIZE)):
        batch = subset.iloc[i:i+BATCH_SIZE]

        outputs = []
        for prompt in batch.prompt.tolist():
            try:
                out = call_llm(MODELS, prompt, temp)
            except Exception as e:
                out = f"[ERROR] {str(e)}"
            outputs.append(out)

        # save batch results
        for r, out in zip(batch.to_dict(orient='records'), outputs):
            records.append({
                "model": MODELS,
                "example_id": r["example_id"],
                "category": r["category"],
                "style": r["prompt_style"],
                "temp": temp,
                "response": out
            })

# save results
responses = pd.DataFrame(records)
output_file = os.path.join(RESULTS_DIR, f"{MODELS}_outputs.csv")
responses.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")
print(f"Total responses collected: {len(responses)}")

# ------------------------
# validation run on heavy models
validation_subset = (
    bbq_df.groupby("category", group_keys=False)
    .apply(lambda x: x.sample(n=10, random_state=123))
    .reset_index(drop=True)
)

print(f"\nRunning validation for model: {MODELS}")
print(f"Validation subset size: {len(validation_subset)} prompts × {len(TEMPERATURES)} temperatures")

val_records = []
for temp in TEMPERATURES:
    print(f"\nProcessing temperature: {temp}")
    for _, row in tqdm(validation_subset.iterrows(), total=len(validation_subset)):
        try:
            out = call_llm(MODELS, row.prompt, temp)
        except Exception as e:
            out = f"[ERROR] {str(e)}"

        val_records.append({
            "model": MODELS,
            "example_id": row["example_id"],
            "category": row["category"],
            "style": row["prompt_style"],
            "temp": temp,
            "response": out
        })

# save reuslts
df_val = pd.DataFrame(val_records)
val_file = os.path.join(RESULTS_DIR, f"{MODELS}_validation.csv")
df_val.to_csv(val_file, index=False)
print(f"Validation results saved to {val_file}")


