from tokenizers import Tokenizer
from pathlib import Path
import numpy as np

ROOT = Path("/data/AlbertLM")

tokenizer = Tokenizer.from_file(
    str(ROOT / "tokenizer/tokenizer.json")
)

input_file = ROOT / "data/processed/corpus.txt"
output_file = ROOT / "data/processed/train.bin"

dtype = np.uint32

count = 0

with open(input_file, "r", encoding="utf-8") as f:
    with open(output_file, "wb") as out:
        for i, line in enumerate(f):
            ids = tokenizer.encode(line).ids
            np.array(ids, dtype=dtype).tofile(out)
            count += len(ids)

            if i % 100000 == 0:
                print(
                    f"lines={i}, tokens={count}"
                )

print("done")
print("tokens:", count)
