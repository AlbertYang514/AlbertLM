#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

ROOT = Path("/data/AlbertLM")
PROCESSED = ROOT / "data" / "processed"
ACTIVE = PROCESSED / "train.bin"
SYNTHETIC = (
    ROOT / "data" / "synthetic" /
    "deepseek-v4-pro" /
    "train-deepseek-v4-pro.bin"
)

TOKENS_PER_UPDATE = 131072
BYTES_PER_TOKEN = 4

checkpoints = sorted(
    ROOT.joinpath("checkpoints").glob("step_*"),
    key=lambda p: int(p.name.rsplit("_", 1)[-1]),
)

if not checkpoints:
    raise RuntimeError("没有找到 checkpoint")

latest = checkpoints[-1]
step = int(latest.name.rsplit("_", 1)[-1])
cut_tokens = step * TOKENS_PER_UPDATE
cut_bytes = cut_tokens * BYTES_PER_TOKEN

old = ACTIVE.resolve()

if not old.exists():
    raise RuntimeError(f"active dataset missing: {old}")
if not SYNTHETIC.exists():
    raise RuntimeError(f"synthetic bin missing: {SYNTHETIC}")
if old == SYNTHETIC:
    raise RuntimeError("active dataset unexpectedly equals synthetic bin")
if old.stat().st_size % BYTES_PER_TOKEN:
    raise RuntimeError("old bin is not uint32")
if SYNTHETIC.stat().st_size % BYTES_PER_TOKEN:
    raise RuntimeError("synthetic bin is not uint32")
if cut_bytes > old.stat().st_size:
    raise RuntimeError("checkpoint cursor exceeds old dataset")

synthetic_tokens = (
    SYNTHETIC.stat().st_size // BYTES_PER_TOKEN
)
old_tokens = old.stat().st_size // BYTES_PER_TOKEN

out = PROCESSED / (
    f"train-final-ds4p-step-{step:010d}.bin"
)
tmp = Path(str(out) + ".tmp")

if out.exists():
    raise RuntimeError(f"output already exists: {out}")

print(f"checkpoint={latest}")
print(f"step={step:,}")
print(f"cut_tokens={cut_tokens:,}")
print(f"old={old}")
print(f"old_tokens={old_tokens:,}")
print(f"synthetic_tokens={synthetic_tokens:,}")
print(f"new_tokens={old_tokens + synthetic_tokens:,}")
print(f"output={out}")

def copy_exact(src, dst, remaining):
    chunk_size = 64 * 1024 * 1024
    while remaining:
        block = src.read(min(chunk_size, remaining))
        if not block:
            raise RuntimeError("unexpected EOF")
        dst.write(block)
        remaining -= len(block)

try:
    with old.open("rb") as old_f, \
         SYNTHETIC.open("rb") as syn_f, \
         tmp.open("wb") as out_f:

        # 保持 checkpoint 之前的数据完全不变
        copy_exact(old_f, out_f, cut_bytes)

        # checkpoint 恢复后立刻读取 DeepSeek 数据
        shutil.copyfileobj(
            syn_f,
            out_f,
            length=64 * 1024 * 1024,
        )

        # 再接回旧数据剩余部分
        shutil.copyfileobj(
            old_f,
            out_f,
            length=64 * 1024 * 1024,
        )

        out_f.flush()
        os.fsync(out_f.fileno())

    os.replace(tmp, out)

    expected = (
        old.stat().st_size +
        SYNTHETIC.stat().st_size
    )
    if out.stat().st_size != expected:
        raise RuntimeError("new bin size mismatch")

    link_tmp = PROCESSED / "train.bin.next"
    link_tmp.unlink(missing_ok=True)
    link_tmp.symlink_to(out.name)
    os.replace(link_tmp, ACTIVE)

    print("active_dataset=", ACTIVE.resolve())
    print("DONE")

finally:
    tmp.unlink(missing_ok=True)
