import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time


OUTPUT = Path("data/raw/web.txt")

URLS = [
    "https://en.wikipedia.org/wiki/Transformer_(machine_learning)",
]


def clean_html(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    for tag in soup(
        ["script","style","nav"]
    ):
        tag.decompose()

    text = soup.get_text(
        "\n"
    )

    lines = [
        x.strip()
        for x in text.splitlines()
        if len(x.strip()) > 20
    ]

    return "\n".join(lines)



def main():

    OUTPUT.parent.mkdir(
        exist_ok=True
    )

    with OUTPUT.open(
        "a",
        encoding="utf-8"
    ) as f:

        for url in URLS:

            print(
                "fetch:",
                url
            )

            r=requests.get(
                url,
                timeout=20,
                headers={
                    "User-Agent":
                    "AlbertLM-CorpusBot/1.0"
                }
            )

            text=clean_html(
                r.text
            )

            f.write(
                text+"\n"
            )

            time.sleep(1)


if __name__=="__main__":
    main()
