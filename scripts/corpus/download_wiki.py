from datasets import load_dataset
from pathlib import Path


def download(lang, out):

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading wikipedia {lang}")

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101." + lang,
        split="train"
    )

    with out.open(
        "w",
        encoding="utf-8"
    ) as f:
        for item in ds:
            text = item.get("text", "").strip()

            if len(text) > 200:
                f.write(text)
                f.write("\n\n")

    print("saved:", out)


if __name__ == "__main__":

    download(
        "zh",
        "data/raw/zh/wiki.txt"
    )

    download(
        "en",
        "data/raw/en/wiki.txt"
    )
