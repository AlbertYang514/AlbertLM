# AlbertLM training operations

Run all production commands from `/data/AlbertLM` with the production Python at
`/data/AlbertLM/.venv/bin/python`.

## Control commands

```bash
scripts/albertlmctl.sh start
scripts/albertlmctl.sh stop
scripts/albertlmctl.sh status
scripts/albertlmctl.sh logs
scripts/albertlmctl.sh gpu
scripts/albertlmctl.sh checkpoints
```

`stop` is graceful. The training signal handler finishes at a safe boundary,
saves a complete DeepSpeed checkpoint, and exits; do not replace it with a
forced kill.

## Production runtime

- Model config: `configs/model/albertlm-1.7b.yaml`
- DeepSpeed config: `configs/deepspeed/albertlm-1.7b-zero2.json`
- Transformer Engine deployment:
  `/data/AlbertLM/.venv/te-deployments/transformer-engine-main-dc4a11dd-7122ebc8`
- Training projections: Transformer Engine FP8 with `DelayedScaling`
- Runtime samples: CUDA BF16 autocast with Transformer Engine FP8 disabled

The periodic hook and one-shot smoke tool both use the public
`AlbertLM.forward(..., fp8_enabled=False)` path. Training calls omit that keyword,
so the configured FP8 default remains active.

## Checkpoint recovery

`checkpoints/latest` is an ordinary UTF-8 tag file, not a symlink. Its tag must
name a complete directory under `/data/AlbertLM/checkpoints`. Never restore or
sample from `step_000000002519`; it is permanently quarantined as contaminated.

Transformer Engine extra-state loading requires
`NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1`. `scripts/start_training.sh` exports the
variable only into the AlbertLM tmux/supervisor process tree. This opt-in is only
for trusted checkpoints created by this project under
`/data/AlbertLM/checkpoints`; never use it for downloaded or otherwise external
checkpoints.

Checkpoints, datasets, tokenized binaries, logs, caches, and local environments
are runtime state and must not be added to Git.

## One-shot sample verification

With training stopped and a clean latest checkpoint:

```bash
NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 \
  /data/AlbertLM/.venv/bin/python tools/smoke_runtime_sample.py
```

The tool loads only the trusted local latest checkpoint, performs no backward or
optimizer/scheduler work, and verifies that the latest tag and checkpoint file
metadata are unchanged.
