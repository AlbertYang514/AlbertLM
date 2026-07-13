import json
from datetime import datetime
from pathlib import Path
from albertlm.checkpoint import save_checkpoint
import torch
from torch.utils.data import Dataset, DataLoader

from albertlm.config import load_config
from albertlm.model import AlbertLM


class ToyDataset(Dataset):

    def __init__(
        self,
        vocab_size,
        seq_len,
        size
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size


    def __len__(self):
        return self.size


    def __getitem__(self, idx):

        start = torch.randint(
            0,
            self.vocab_size - self.seq_len - 1,
            (1,)
        ).item()


        tokens = torch.arange(
            start,
            start + self.seq_len
        )

        return tokens



def write_status(
    status,
    step,
    loss,
    checkpoint=None
):

    data = {
        "time": datetime.now().isoformat(),
        "status": status,
        "step": step,
        "loss": float(loss),
        "checkpoint": checkpoint,
        "gpu": "RTX 5090D v2"
    }


    Path("logs").mkdir(
        exist_ok=True
    )


    with open(
        "logs/status.json",
        "w"
    ) as f:
        json.dump(
            data,
            f,
            indent=2
        )



def main():

    device="cuda"

    config = load_config(
        "configs/model/albertlm-125m.yaml"
    )


    model = AlbertLM(config)

    model = (
        model
        .cuda()
        .bfloat16()
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1
    )


    dataset = ToyDataset(
        config.vocab_size,
        128,
        10000
    )


    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )


    model.train()

    for step, batch in enumerate(loader):

        batch = batch.cuda()


        out = model(
            batch,
            labels=batch
        )

        loss = out["loss"]


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        if step % 10 == 0:

            print(
                f"step {step} loss {loss.item():.4f}"
            )

            write_status(
                "training",
                step,
                loss.item()
            )


        if step % 100 == 0 and step > 0:

            checkpoint_path = (
                f"checkpoints/step_{step}.pt"
            )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step
            )

            write_status(
                "training",
                step,
                loss.item(),
                checkpoint_path
            )

            print(
                f"checkpoint saved: step {step}"
            )


        if step >= 1000000:
            break


if __name__=="__main__":
    main()

