import json
from pathlib import Path

DATA_DIR   = Path("data/raw")
PROMPT_DIR = Path("prompts")

def preprocess_bbq(data_dir: Path = DATA_DIR, prompt_dir: Path = PROMPT_DIR):
    prompt_dir.mkdir(exist_ok=True, parents=True)

    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {data_dir.resolve()}")
        return

    for file_path in files:
        category = file_path.stem  # e.g. "age", "gender", etc.
        conv_prompts = []
        gen_prompts  = []

        with open(file_path, "r") as f:
            lines = f.readlines()
        print(f"Processing {file_path.name}: {len(lines)} entries")

        for line in lines:
            entry = json.loads(line)

            # extract exactly these four fields
            ctx      = entry["context"]
            cond     = entry["context_condition"]  # "ambig" or "disambig"
            question = entry["question"]
            choices  = [entry["ans0"], entry["ans1"], entry["ans2"]]

            opts_str = ", ".join(choices)

            # ---- conversational prompt ----
            conv = (
                f"Context ({cond}): {ctx}\n"
                f"Question: {question}\n"
                f"Options: {opts_str}\n"
                "Answer:"
            )
            conv_prompts.append(conv)

            # ---- generative prompt ----
            gen = (
                f"Given the following {cond} context: {ctx}. "
                f"Please provide your reasoning and answer to: {question}. "
                f"Options are {opts_str}. "
                "Respond with your reasoning then final answer."
            )
            gen_prompts.append(gen)

        # write out one file per style
        with open(prompt_dir / f"{category}_conversational.json", "w") as out_f:
            json.dump(conv_prompts, out_f, indent=2)
        with open(prompt_dir / f"{category}_generative.json", "w") as out_f:
            json.dump(gen_prompts, out_f, indent=2)

        print(f"  → wrote {len(conv_prompts)} conversational and {len(gen_prompts)} generative prompts")

    print(f"\nDone! All prompts are in {prompt_dir.resolve()}")

if __name__ == "__main__":
    preprocess_bbq()

