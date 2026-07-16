#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import os
import random
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xxhash
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer


ROOT = Path("/data/AlbertLM")
TOKENIZER_PATH = ROOT / "tokenizer/tokenizer.json"

BUILD_DIR = ROOT / "data/build/pretrain-10b"
SOURCE_DIR = BUILD_DIR / "sources"
PROCESSED_DIR = ROOT / "data/processed"
STATS_DIR = ROOT / "data/stats"

SEEN_PATH = BUILD_DIR / "seen.u64"
LOCK_PATH = (
    ROOT
    / "data/manifests/pretrain-10b-resolved.json"
)
PROVENANCE_PATH = (
    ROOT
    / "data/manifests/code-provenance.jsonl.gz"
)

SEQ_LEN = 2048
VALIDATION_MODULUS = 200
SEED = 20260714

ENCODE_BATCH_DOCS = 128
SHUFFLE_BUFFER = 20_000
MIX_BUFFER_BLOCKS = 8192
LOG_EVERY_TOKENS = 10_000_000

SOURCE_TARGETS = OrderedDict(
    [
        ("fineweb-edu-en", 3_000_006_656),
        ("fineweb2-zh", 2_999_998_464),
        ("fineweb2-ja", 999_999_488),
        ("wiki-en", 750_000_000),
        ("wiki-zh", 389_000_000),
        ("code-python", 450_000_896),
        ("code-javascript", 199_999_488),
        ("code-typescript", 149_999_616),
        ("code-java", 149_999_616),
        ("code-cpp", 149_999_616),
        ("code-go", 99_999_744),
        ("code-rust", 99_999_744),
        ("code-shell", 49_999_872),
        ("code-c", 99_999_744),
        ("code-sql", 49_999_872),
    ]
)

CODE_DATA_DIRS = {
    "code-python": "data/python",
    "code-javascript": "data/javascript",
    "code-typescript": "data/typescript",
    "code-java": "data/java",
    "code-cpp": "data/cpp",
    "code-go": "data/go",
    "code-rust": "data/rust",
    "code-shell": "data/shell",
    "code-c": "data/c",
    "code-sql": "data/sql",
}

REPOSITORIES = (
    "HuggingFaceFW/fineweb-edu",
    "HuggingFaceFW/fineweb-2",
    "bigcode/the-stack-dedup",
)


def log(message):
    print(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        message,
        flush=True,
    )


def tokens_in_file(path):
    if not path.exists():
        return 0

    size = path.stat().st_size

    if size % 4:
        with path.open("r+b") as file:
            file.truncate(size - size % 4)

        size = path.stat().st_size

    return size // 4


class BinaryWriter:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.tokens = tokens_in_file(path)

        self.file = path.open(
            "ab",
            buffering=8 * 1024 * 1024,
        )

    def write(self, token_ids, limit=None):
        array = np.asarray(
            token_ids,
            dtype=np.uint32,
        )

        if limit is not None:
            array = array[
                :max(0, int(limit))
            ]

        if len(array):
            array.tofile(self.file)
            self.tokens += len(array)

    def flush(self):
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.flush()
        self.file.close()


def train_path(name):
    return (
        SOURCE_DIR
        / f"{name}.train.bin"
    )


def validation_path(name):
    return (
        SOURCE_DIR
        / f"{name}.validation.bin"
    )


def normalize_text(value):
    if not isinstance(value, str):
        return None

    text = (
        value
        .replace("\x00", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = unicodedata.normalize(
        "NFC",
        text,
    ).strip()

    if len(text) < 80:
        return None

    if len(text) > 250_000:
        text = text[:250_000]

    visible = sum(
        not character.isspace()
        for character in text
    )

    if visible < 40:
        return None

    return text


def local_line_iterator(
    path,
    seed,
):
    rng = random.Random(seed)
    buffer = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for line in file:
            if line.strip():
                buffer.append(line)

            if len(buffer) >= 100_000:
                rng.shuffle(buffer)

                for item in buffer:
                    yield {
                        "text": item,
                    }

                buffer.clear()

        rng.shuffle(buffer)

        for item in buffer:
            yield {
                "text": item,
            }


def resolve_revisions():
    api = HfApi()
    revisions = {}

    for repository in REPOSITORIES:
        info = api.dataset_info(
            repository
        )

        revisions[repository] = (
            info.sha
        )

        log(
            f"locked {repository} "
            f"to {info.sha}"
        )

    LOCK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOCK_PATH.write_text(
        json.dumps(
            revisions,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return revisions


def hf_stream(
    repository,
    revision,
    seed,
    name=None,
    data_dir=None,
):
    arguments = {
        "path": repository,
        "revision": revision,
        "split": "train",
        "streaming": True,
    }

    if name is not None:
        arguments["name"] = name

    if data_dir is not None:
        arguments["data_dir"] = (
            data_dir
        )

    dataset = load_dataset(
        **arguments
    )

    return dataset.shuffle(
        seed=seed,
        buffer_size=SHUFFLE_BUFFER,
    )


def source_specs(revisions):
    yield (
        "fineweb-edu-en",
        hf_stream(
            "HuggingFaceFW/fineweb-edu",
            revisions[
                "HuggingFaceFW/fineweb-edu"
            ],
            SEED + 1,
            name="sample-10BT",
        ),
        "text",
    )

    yield (
        "fineweb2-zh",
        hf_stream(
            "HuggingFaceFW/fineweb-2",
            revisions[
                "HuggingFaceFW/fineweb-2"
            ],
            SEED + 2,
            name="cmn_Hani",
        ),
        "text",
    )

    yield (
        "fineweb2-ja",
        hf_stream(
            "HuggingFaceFW/fineweb-2",
            revisions[
                "HuggingFaceFW/fineweb-2"
            ],
            SEED + 3,
            name="jpn_Jpan",
        ),
        "text",
    )

    yield (
        "wiki-en",
        local_line_iterator(
            PROCESSED_DIR / "en.txt",
            SEED + 4,
        ),
        "text",
    )

    yield (
        "wiki-zh",
        local_line_iterator(
            PROCESSED_DIR / "zh.txt",
            SEED + 5,
        ),
        "text",
    )

    for offset, (
        source_name,
        data_dir,
    ) in enumerate(
        CODE_DATA_DIRS.items(),
        start=10,
    ):
        yield (
            source_name,
            hf_stream(
                "bigcode/the-stack-dedup",
                revisions[
                    "bigcode/the-stack-dedup"
                ],
                SEED + offset,
                data_dir=data_dir,
            ),
            "content",
        )


def load_seen_hashes():
    BUILD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SEEN_PATH.exists():
        values = np.fromfile(
            SEEN_PATH,
            dtype=np.uint64,
        )

        seen = set(
            map(int, values)
        )

        log(
            "loaded exact-dedup "
            f"hashes: {len(seen):,}"
        )
    else:
        seen = set()

    handle = SEEN_PATH.open(
        "ab",
        buffering=4 * 1024 * 1024,
    )

    return seen, handle


def process_source(
    name,
    stream,
    text_field,
    target_tokens,
    tokenizer,
    eos_id,
    seen,
    seen_handle,
    provenance_handle,
):
    train_writer = BinaryWriter(
        train_path(name)
    )

    validation_writer = BinaryWriter(
        validation_path(name)
    )

    if train_writer.tokens >= target_tokens:
        if train_writer.tokens > target_tokens:
            train_writer.close()

            with train_path(name).open(
                "r+b"
            ) as file:
                file.truncate(
                    target_tokens * 4
                )

            validation_writer.close()
        else:
            train_writer.close()
            validation_writer.close()

        log(
            f"{name}: already complete"
        )
        return

    log(
        f"{name}: resume "
        f"{train_writer.tokens:,}/"
        f"{target_tokens:,}"
    )

    pending_texts = []
    pending_hashes = []
    pending_metadata = []

    scanned = 0
    accepted = 0

    next_log = (
        train_writer.tokens
        // LOG_EVERY_TOKENS
        + 1
    ) * LOG_EVERY_TOKENS

    def flush_batch():
        nonlocal accepted
        nonlocal next_log

        encodings = (
            tokenizer.encode_batch(
                pending_texts,
                add_special_tokens=False,
            )
        )

        for (
            encoding,
            document_hash,
            metadata,
        ) in zip(
            encodings,
            pending_hashes,
            pending_metadata,
        ):
            if (
                train_writer.tokens
                >= target_tokens
            ):
                return True

            ids = encoding.ids

            if not ids:
                continue

            ids = ids + [eos_id]

            seen.add(document_hash)

            np.asarray(
                [document_hash],
                dtype=np.uint64,
            ).tofile(seen_handle)

            accepted += 1

            if metadata:
                provenance_handle.write(
                    json.dumps(
                        {
                            "hash_xxh3_64":
                                f"{document_hash:016x}",
                            "source": name,
                            **metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if (
                document_hash
                % VALIDATION_MODULUS
                == 0
            ):
                validation_writer.write(
                    ids
                )
            else:
                train_writer.write(
                    ids,
                    target_tokens
                    - train_writer.tokens,
                )

            if (
                train_writer.tokens
                >= next_log
            ):
                train_writer.flush()
                validation_writer.flush()
                seen_handle.flush()
                provenance_handle.flush()

                os.fsync(
                    seen_handle.fileno()
                )

                log(
                    f"{name}: "
                    f"{train_writer.tokens:,}/"
                    f"{target_tokens:,} train; "
                    f"{validation_writer.tokens:,} "
                    f"validation; "
                    f"{scanned:,} scanned; "
                    f"{accepted:,} accepted"
                )

                next_log += (
                    LOG_EVERY_TOKENS
                )

        return (
            train_writer.tokens
            >= target_tokens
        )

    try:
        for sample in stream:
            scanned += 1

            text = normalize_text(
                sample.get(text_field)
            )

            if text is None:
                continue

            document_hash = (
                xxhash
                .xxh3_64_intdigest(text)
            )

            if document_hash in seen:
                continue

            pending_texts.append(text)
            pending_hashes.append(
                document_hash
            )

            if name.startswith(
                "code-"
            ):
                metadata = {
                    key: sample.get(key)
                    for key in (
                        "hexsha",
                        "lang",
                        "ext",
                        "max_stars_repo_name",
                        "max_stars_repo_licenses",
                    )
                    if sample.get(key)
                    is not None
                }
            else:
                metadata = {}

            pending_metadata.append(
                metadata
            )

            if (
                len(pending_texts)
                >= ENCODE_BATCH_DOCS
            ):
                complete = flush_batch()

                pending_texts.clear()
                pending_hashes.clear()
                pending_metadata.clear()

                if complete:
                    break

        if (
            pending_texts
            and train_writer.tokens
            < target_tokens
        ):
            flush_batch()

        if (
            train_writer.tokens
            < target_tokens
        ):
            raise RuntimeError(
                f"{name} exhausted at "
                f"{train_writer.tokens:,}/"
                f"{target_tokens:,}"
            )

        train_writer.flush()
        validation_writer.flush()
        seen_handle.flush()
        provenance_handle.flush()

        os.fsync(
            seen_handle.fileno()
        )

        done_path = (
            SOURCE_DIR
            / f"{name}.done"
        )

        done_path.write_text(
            json.dumps(
                {
                    "source": name,
                    "train_tokens":
                        train_writer.tokens,
                    "validation_tokens":
                        validation_writer.tokens,
                    "target_tokens":
                        target_tokens,
                    "scanned_this_run":
                        scanned,
                    "accepted_this_run":
                        accepted,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        log(
            f"{name}: COMPLETE"
        )

    finally:
        train_writer.close()
        validation_writer.close()


def allocate_source_counts(
    remaining,
    count,
):
    total = int(
        remaining.sum()
    )

    if count >= total:
        return remaining.copy()

    exact = (
        remaining.astype(np.float64)
        * count
        / total
    )

    allocation = np.minimum(
        np.floor(exact).astype(
            np.int64
        ),
        remaining,
    )

    missing = (
        count
        - int(allocation.sum())
    )

    for index in np.argsort(
        -(exact - allocation)
    ):
        if missing == 0:
            break

        room = int(
            remaining[index]
            - allocation[index]
        )

        take = min(
            room,
            missing,
        )

        allocation[index] += take
        missing -= take

    if missing:
        raise RuntimeError(
            "mix allocation failed"
        )

    return allocation


def mix_files(
    input_paths,
    output_path,
    seed,
):
    arrays = []
    block_counts = []

    for path in input_paths:
        blocks = (
            tokens_in_file(path)
            // SEQ_LEN
        )

        if blocks:
            arrays.append(
                np.memmap(
                    path,
                    dtype=np.uint32,
                    mode="r",
                )
            )

            block_counts.append(
                blocks
            )

    if not arrays:
        raise RuntimeError(
            f"no input blocks for "
            f"{output_path}"
        )

    remaining = np.asarray(
        block_counts,
        dtype=np.int64,
    )

    positions = np.zeros(
        len(arrays),
        dtype=np.int64,
    )

    total_blocks = int(
        remaining.sum()
    )

    written_blocks = 0

    rng = np.random.default_rng(
        seed
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix + ".tmp"
        )
    )

    temporary_path.unlink(
        missing_ok=True
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with temporary_path.open(
        "wb",
        buffering=16 * 1024 * 1024,
    ) as output:
        while (
            written_blocks
            < total_blocks
        ):
            batch_blocks = min(
                MIX_BUFFER_BLOCKS,
                total_blocks
                - written_blocks,
            )

            allocation = (
                allocate_source_counts(
                    remaining,
                    batch_blocks,
                )
            )

            source_ids = np.repeat(
                np.arange(
                    len(arrays),
                    dtype=np.int16,
                ),
                allocation,
            )

            rng.shuffle(
                source_ids
            )

            buffer = np.empty(
                (
                    batch_blocks,
                    SEQ_LEN,
                ),
                dtype=np.uint32,
            )

            for row, raw_source_id in (
                enumerate(source_ids)
            ):
                source_id = int(
                    raw_source_id
                )

                start = (
                    int(
                        positions[
                            source_id
                        ]
                    )
                    * SEQ_LEN
                )

                buffer[row] = (
                    arrays[source_id][
                        start:
                        start + SEQ_LEN
                    ]
                )

                positions[
                    source_id
                ] += 1

                remaining[
                    source_id
                ] -= 1

            buffer.tofile(output)

            written_blocks += (
                batch_blocks
            )

            if (
                written_blocks
                % (
                    MIX_BUFFER_BLOCKS
                    * 10
                )
                == 0
            ):
                output.flush()

                log(
                    f"mix "
                    f"{output_path.name}: "
                    f"{written_blocks:,}/"
                    f"{total_blocks:,} blocks"
                )

        output.flush()
        os.fsync(output.fileno())

    os.replace(
        temporary_path,
        output_path,
    )

    output_tokens = (
        written_blocks
        * SEQ_LEN
    )

    log(
        f"mix COMPLETE "
        f"{output_path}: "
        f"{output_tokens:,} tokens"
    )

    return output_tokens


def activate_training_data(
    train_output,
):
    active_path = (
        PROCESSED_DIR
        / "train.bin"
    )

    old_backup = (
        PROCESSED_DIR
        / "train-wiki-4.69b.bin"
    )

    if active_path.is_symlink():
        active_path.unlink()

    elif active_path.exists():
        if not old_backup.exists():
            os.replace(
                active_path,
                old_backup,
            )
        else:
            active_path.unlink()

    active_path.symlink_to(
        train_output.name
    )

    log(
        f"active dataset: "
        f"{active_path} -> "
        f"{train_output.name}"
    )


def main():
    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "true",
    )

    for directory in (
        SOURCE_DIR,
        PROCESSED_DIR,
        STATS_DIR,
        LOCK_PATH.parent,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    for path in (
        TOKENIZER_PATH,
        PROCESSED_DIR / "en.txt",
        PROCESSED_DIR / "zh.txt",
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )


    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    eos_id = tokenizer.token_to_id(
        "<eos>"
    )

    if eos_id is None:
        raise RuntimeError(
            "tokenizer has no <eos>"
        )

    revisions = (
        resolve_revisions()
    )

    seen, seen_handle = (
        load_seen_hashes()
    )

    provenance_handle = gzip.open(
        PROVENANCE_PATH,
        "at",
        encoding="utf-8",
        compresslevel=6,
    )

    try:
        for (
            source_name,
            stream,
            text_field,
        ) in source_specs(revisions):
            target = SOURCE_TARGETS[
                source_name
            ]

            if (
                tokens_in_file(
                    train_path(
                        source_name
                    )
                )
                >= target
            ):
                log(
                    f"{source_name}: skip"
                )
                continue

            process_source(
                source_name,
                stream,
                text_field,
                target,
                tokenizer,
                eos_id,
                seen,
                seen_handle,
                provenance_handle,
            )

    finally:
        seen_handle.flush()
        os.fsync(
            seen_handle.fileno()
        )
        seen_handle.close()
        provenance_handle.close()

    train_output = (
        PROCESSED_DIR
        / "train-10b.bin"
    )

    validation_output = (
        PROCESSED_DIR
        / "validation-10b.bin"
    )

    train_tokens = mix_files(
        [
            train_path(name)
            for name
            in SOURCE_TARGETS
        ],
        train_output,
        SEED + 1000,
    )

    validation_tokens = mix_files(
        [
            validation_path(name)
            for name
            in SOURCE_TARGETS
        ],
        validation_output,
        SEED + 2000,
    )

    activate_training_data(
        train_output
    )

    stats = {
        "name":
            "albertlm-pretrain-10b",
        "created_at":
            time.strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
        "sequence_length":
            SEQ_LEN,
        "train_path":
            str(train_output),
        "validation_path":
            str(validation_output),
        "train_tokens":
            train_tokens,
        "validation_tokens":
            validation_tokens,
        "train_blocks":
            train_tokens // SEQ_LEN,
        "validation_blocks":
            validation_tokens
            // SEQ_LEN,
        "sources": {
            name: {
                "train_tokens":
                    tokens_in_file(
                        train_path(name)
                    ),
                "validation_tokens":
                    tokens_in_file(
                        validation_path(
                            name
                        )
                    ),
            }
            for name
            in SOURCE_TARGETS
        },
        "revisions":
            revisions,
        "exact_dedup_hashes":
            len(seen),
    }

    stats_path = (
        STATS_DIR
        / "pretrain-10b.json"
    )

    stats_path.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    log(
        "ALL COMPLETE; "
        f"stats={stats_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
