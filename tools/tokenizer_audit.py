#!/usr/bin/env python3
import json, math, time, unicodedata
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT = Path('/data/AlbertLM')
TOKENIZER = ROOT / 'tokenizer/tokenizer.json'
RAW = ROOT / 'data/synthetic/deepseek-v4-pro/raw.jsonl'
BIN = ROOT / 'data/processed/train-final.bin'
REPORT = ROOT / 'reports/tokenizer-audit.json'
EXPECTED_TOKENS = 14_331_990_016
SAMPLE_TOKENS = 100_000_000
CHUNKS = 100


def summary(counts, name, special_ids):
    used = counts > 0
    normal = np.ones(len(counts), dtype=bool)
    normal[list(special_ids)] = False
    total = int(counts.sum())
    sorted_counts = np.sort(counts)[::-1]
    top1n = max(1, math.ceil(len(counts) * 0.01))
    top10n = max(1, math.ceil(len(counts) * 0.10))
    out = {
        'name': name,
        'sampled_tokens': total,
        'used': int(used.sum()),
        'utilization_pct': float(used.mean() * 100),
        'normal_used': int((used & normal).sum()),
        'normal_utilization_pct': float((used & normal).sum() / normal.sum() * 100),
        'unused': int((counts == 0).sum()),
        'once': int((counts == 1).sum()),
        '2_to_10': int(((counts >= 2) & (counts <= 10)).sum()),
        '11_to_100': int(((counts >= 11) & (counts <= 100)).sum()),
        '101_to_1000': int(((counts >= 101) & (counts <= 1000)).sum()),
        'over_1000': int((counts > 1000).sum()),
        'top_1pct_share': float(sorted_counts[:top1n].sum() / total) if total else 0,
        'top_10pct_share': float(sorted_counts[:top10n].sum() / total) if total else 0,
    }
    print(f"\n=== {name} ===")
    for k, v in out.items():
        if k != 'name':
            print(f'{k}: {v:,}' if isinstance(v, int) else f'{k}: {v:.4f}')
    return out


def audit_raw(tok, vocab_size):
    counts = np.zeros(vocab_size, dtype=np.uint64)
    docs = malformed = exact = nfc = 0
    mismatches = []
    with RAW.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            text = row.get('output')
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False).ids
            decoded = tok.decode(ids, skip_special_tokens=False)
            normalized = unicodedata.normalize('NFC', text)
            if decoded == text:
                exact += 1
            elif decoded == normalized:
                nfc += 1
            elif len(mismatches) < 10:
                mismatches.append({'line': line_no, 'id': row.get('id')})
            counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab_size).astype(np.uint64)
            docs += 1
            if docs % 500 == 0:
                print(f'raw_progress={docs:,}', flush=True)
    return counts, {
        'documents': docs,
        'malformed_or_partial_lines': malformed,
        'exact_roundtrip_ok': exact,
        'nfc_roundtrip_ok': nfc,
        'roundtrip_failures': docs - exact - nfc,
        'mismatch_examples': mismatches,
    }


def audit_bin(vocab_size):
    size = BIN.stat().st_size
    if size == EXPECTED_TOKENS * 2:
        dtype = np.uint16
    elif size == EXPECTED_TOKENS * 4:
        dtype = np.uint32
    else:
        raise RuntimeError(f'bin大小与已知token数不匹配: bytes={size:,}')
    mm = np.memmap(BIN, dtype=dtype, mode='r')
    chunk_len = SAMPLE_TOKENS // CHUNKS
    starts = np.linspace(0, len(mm) - chunk_len, CHUNKS, dtype=np.int64)
    counts = np.zeros(vocab_size, dtype=np.uint64)
    invalid = 0
    scanned = 0
    for i, start in enumerate(starts, 1):
        arr = np.asarray(mm[start:start + chunk_len], dtype=np.int64)
        valid = arr < vocab_size
        invalid += int((~valid).sum())
        counts += np.bincount(arr[valid], minlength=vocab_size).astype(np.uint64)
        scanned += len(arr)
        if i % 10 == 0:
            print(f'bin_progress={i}/{CHUNKS} scanned={scanned:,}', flush=True)
    return counts, {
        'dtype': np.dtype(dtype).name,
        'file_tokens': len(mm),
        'sampled_tokens': scanned,
        'invalid_ids': invalid,
    }


def adversarial(tok):
    tests = [
        '中文\n第二行\t缩进', '日本語とカタカナ', 'English   spaces\r\nnext',
        'é e\u0301 Å A\u030A', '👁️👄👁️ 👨‍👩‍👧‍👦 🏳️‍🌈',
        '```python\ndef f(x):\n    return x**2\n```',
        '{"路径":"/data/AlbertLM","a":[1,null]}',
        '𠮷 𓀀 𝄞 \u200bzero\u200dwidth', 'nul:\x00:end',
    ]
    bad = []
    for i, text in enumerate(tests):
        decoded = tok.decode(tok.encode(text, add_special_tokens=False).ids,
                             skip_special_tokens=False)
        if decoded != unicodedata.normalize('NFC', text):
            bad.append(i)
    return {'cases': len(tests), 'failures': len(bad), 'failed_cases': bad}


def main():
    started = time.time()
    tok = Tokenizer.from_file(str(TOKENIZER))
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    cfg = json.loads(TOKENIZER.read_text(encoding='utf-8'))
    special_ids = {int(x['id']) for x in cfg.get('added_tokens', [])
                   if x.get('special') and 'id' in x}

    raw_counts, raw_meta = audit_raw(tok, vocab_size)
    bin_counts, bin_meta = audit_bin(vocab_size)
    report = {
        'tokenizer': str(TOKENIZER),
        'vocab_size': vocab_size,
        'special_ids': sorted(special_ids),
        'adversarial_roundtrip': adversarial(tok),
        'raw': {'meta': raw_meta,
                'vocab': summary(raw_counts, 'DeepSeek raw outputs', special_ids)},
        'bin': {'meta': bin_meta,
                'vocab': summary(bin_counts, 'train-final.bin 100M sample', special_ids)},
        'elapsed_seconds': time.time() - started,
    }
    print('\nroundtrip:', json.dumps(raw_meta, ensure_ascii=False))
    print('adversarial:', json.dumps(report['adversarial_roundtrip'], ensure_ascii=False))
    print('bin_meta:', json.dumps(bin_meta, ensure_ascii=False))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nreport={REPORT}')
    print(f"elapsed={report['elapsed_seconds']:.1f}s")


if __name__ == '__main__':
    main()
