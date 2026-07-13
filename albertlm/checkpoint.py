import torch
from pathlib import Path


def save_checkpoint(
    path,
    model,
    optimizer,
    step
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    device="cuda"
):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    return checkpoint["step"]
