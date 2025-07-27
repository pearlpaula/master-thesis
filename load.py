import io
import requests
import pandas as pd
from config import BBQ_BASE_URL, BBQ_FILES

def load_bbq():
    dfs = []
    for fname in BBQ_FILES:
        r = requests.get(BBQ_BASE_URL + fname)
        df = pd.read_json(io.StringIO(r.text), lines=True)
        df['category'] = fname.replace(".jsonl","")
        df['context_length'] = df['context'].str.split().apply(len)
        df['question_length'] = df['question'].str.split().apply(len)
        dfs.append(df)
    bbq = pd.concat(dfs, ignore_index=True)
    return bbq

if __name__ == "__main__":
    df = load_bbq()
    print("Loaded BBQ dataset with shape:", df.shape)
    df.to_parquet("results/bbq_raw.parquet", index=False)
