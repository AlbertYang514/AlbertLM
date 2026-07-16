from pathlib import Path


OUT=Path(
    "data/processed/corpus.txt"
)


INPUTS=list(
    Path("data/processed").glob("*.txt")
)


with OUT.open(
    "w",
    encoding="utf-8"
) as fout:

    for p in INPUTS:

        if p.name=="corpus.txt":
            continue

        print("merge",p)

        with p.open(
            encoding="utf-8"
        ) as f:

            for line in f:
                fout.write(line)


print("saved",OUT)
