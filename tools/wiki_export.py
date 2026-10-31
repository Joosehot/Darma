"""Simple English Wikipedia (Hugging Face wikimedia/wikipedia 20231101.simple, already on disk) -> the lead of every
article: its title and its first sentences, one article per line (title TAB sentence TAB sentence), for Darma to read.

python tools/wiki_export.py [sentences per article, default 2]
"""
import glob
import os
import re
import sys

import pyarrow.parquet as pq

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
src = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--wikimedia--wikipedia/snapshots/*/20231101.simple/*.parquet"))
if not src:
    sys.exit("Simple English Wikipedia not found in the Hugging Face cache")
os.makedirs("data/wiki", exist_ok=True)
# a sentence ends at . ! or ? followed by a space and a capital (not after a single capital: "J. R. R.")
end = re.compile(r"(?<=[.!?])\s+(?=[A-Z])(?<![A-Z]\.\s)")
table = pq.read_table(src[0], columns=["title", "text"])
n = 0
with open("data/wiki/simple_lead.tsv", "w", encoding="utf-8") as out:
    for batch in table.to_batches(10000):
        for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
            first = next((p.strip() for p in text.split("\n") if len(p.strip()) > 20), "")
            ss = [s.strip().replace("\t", " ") for s in end.split(first) if s.strip()][:N]
            if ss:
                out.write(title.replace("\t", " ") + "\t" + "\t".join(ss) + "\n")
                n += 1
print(f"{n} articles -> data/wiki/simple_lead.tsv")
