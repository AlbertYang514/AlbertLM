import json
from pathlib import Path
from datetime import datetime


STATUS_FILE = Path("logs/status.json")


def write_status(**kwargs):

    data = {
        "time": datetime.now().isoformat(),
        **kwargs
    }

    STATUS_FILE.parent.mkdir(
        exist_ok=True
    )

    with open(
        STATUS_FILE,
        "w"
    ) as f:
        json.dump(
            data,
            f,
            indent=2
        )
