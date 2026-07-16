from pathlib import Path
import re


def clean_file(src, dst):

    src = Path(src)
    dst = Path(dst)

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with src.open(
        encoding="utf-8"
    ) as fin, dst.open(
        "w",
        encoding="utf-8"
    ) as fout:

        for line in fin:

            line=line.strip()

            if len(line)<20:
                continue

            line=re.sub(
                r"\s+",
                " ",
                line
            )

            fout.write(line+"\n")


if __name__=="__main__":

    for src,dst in [
        (
            "data/raw/zh/wiki.txt",
            "data/processed/zh.txt"
        ),
        (
            "data/raw/en/wiki.txt",
            "data/processed/en.txt"
        )
    ]:

        clean_file(src,dst)
        print("cleaned",src)
