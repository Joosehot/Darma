"""DailyDialog (the Hugging Face daily_dialog parquet files, put in data/raw/dailydialog/{train,validation,test}.parquet)
-> the JSON Darma reads: {split}.json (the dialogues) and {split}_marked.json (with each turn's act and emotion).

python tools/dailydialog_export.py
"""
import json

import pyarrow.parquet as pq

for split in ["train", "validation", "test"]:
    rows = pq.read_table(f"data/raw/dailydialog/{split}.parquet").to_pylist()
    with open(f"data/raw/dailydialog/{split}.json", "w", encoding="utf-8") as f:
        json.dump([r["dialog"] for r in rows], f)
    with open(f"data/raw/dailydialog/{split}_marked.json", "w", encoding="utf-8") as f:
        json.dump([{"dialog": r["dialog"], "act": r["act"], "emotion": r["emotion"]} for r in rows], f)
    print(split, len(rows), "dialogues")
