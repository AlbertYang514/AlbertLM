import os
from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        temporary_path,
    )

    # 同一文件系统内原子替换，防止写到一半留下损坏 checkpoint。
    os.replace(
        temporary_path,
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    device="cuda",
):
    path = Path(path)

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    return int(
        checkpoint["step"]
    )
