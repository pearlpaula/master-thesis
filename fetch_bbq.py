import requests, pathlib
from config import BBQ_BASE_URL, BBQ_FILES

raw_dir = pathlib.Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

for fname in BBQ_FILES:
    print("Downloading", fname)
    r = requests.get(BBQ_BASE_URL + fname)
    r.raise_for_status()
    (raw_dir / fname).write_text(r.text)
print("Fetched all BBQ JSONLs into data/raw/")
