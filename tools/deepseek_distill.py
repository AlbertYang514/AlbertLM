#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path('/data/AlbertLM')
WORK = ROOT / 'data' / 'synthetic' / 'deepseek-v4-pro'
JOBS = WORK / 'jobs.jsonl'
RAW = WORK / 'raw.jsonl'
ERRORS = WORK / 'errors.jsonl'
CLEAN = WORK / 'clean.jsonl'
CORPUS = WORK / 'corpus.txt'
BIN = WORK / 'train-deepseek-v4-pro.bin'
STATS = WORK / 'stats.json'

SYSTEM_PROMPT = '''你正在为通用语言模型生成高质量合成预训练语料。只输出最终正文，不要提到提示词、数据集、蒸馏、AI身份或生成过程。正文必须自洽、准确、信息密度高；不确定的具体事实不要编造。需要推导时给出可公开阅读的解释和关键步骤，但不要输出隐藏思维过程。代码必须完整且有解释，数学公式必须定义符号。避免空泛套话、营销文案、重复段落和机械总结。'''

CATEGORIES: dict[str, list[str]] = {
    'science': ['物理学', '化学', '生物学', '天文学', '地球科学', '环境科学', '材料科学', '神经科学'],
    'mathematics': ['代数', '概率统计', '微积分', '离散数学', '线性代数', '数论', '优化方法', '信息论'],
    'software': ['Python', '操作系统', '计算机网络', '数据库', '编译原理', '分布式系统', '软件工程', '网络安全', '机器学习系统', '高性能计算'],
    'engineering': ['电子工程', '控制理论', '通信工程', '机械工程', '能源系统', '土木工程', '工业设计', '可靠性工程'],
    'humanities': ['世界史', '中国史', '哲学', '语言学', '文学理论', '艺术史', '宗教学研究', '考古学'],
    'social_science': ['经济学', '社会学', '政治学', '法学基础', '心理学', '公共管理', '传播学', '国际关系'],
    'practical': ['个人财务基础', '项目管理', '科学写作', '信息检索', '批判性思维', '实验设计', '数据分析', '职业沟通'],
    'creative': ['叙事结构', '人物塑造', '场景描写', '科幻设定', '悬疑结构', '散文写作', '诗歌鉴赏', '跨文化写作'],
}

ANGLES = [
    '核心概念与常见误区', '从基本原理到实际应用', '一个具体问题的系统分析', '历史演变与现代方法',
    '关键机制、限制与权衡', '面向初学者但保持技术严谨', '面向工程实践的完整说明', '用例子解释抽象概念',
    '比较两种主流方法并说明适用边界', '给出可验证的推导或实验思路', '从失败案例反推正确方法', '整理为结构清晰的参考文章',
]

LANGUAGES = ['zh-CN'] * 12 + ['en'] * 5 + ['ja'] * 2


def write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f'跳过损坏 JSONL 行: {path}:{line_no}', file=sys.stderr)


def make_jobs(args: argparse.Namespace) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    categories = list(CATEGORIES)
    with JOBS.open('w', encoding='utf-8') as f:
        for i in range(args.count):
            category = categories[i % len(categories)]
            topic = rng.choice(CATEGORIES[category])
            angle = rng.choice(ANGLES)
            language = rng.choice(LANGUAGES)
            diversity_key = hashlib.sha256(f'{args.seed}:{i}'.encode()).hexdigest()[:12]
            if language == 'zh-CN':
                request = (
                    f'写一篇约1200至2200字的高质量参考正文。领域：{category}；主题范围：{topic}；'
                    f'切入角度：{angle}；多样性键：{diversity_key}。请选择一个具体且不过度常见的子题，'
                    '正文应自成一体，包含必要定义、机制、例子和边界条件。不要写问答格式，不要写“下面将”。'
                )
            elif language == 'en':
                request = (
                    f'Write a self-contained high-quality reference article of roughly 900-1600 words. '
                    f'Domain: {category}; topic area: {topic}; angle: {angle}; diversity key: {diversity_key}. '
                    'Choose a specific, non-obvious subtopic. Include definitions, mechanisms, examples, limitations, '
                    'and practical implications. Do not use a Q&A format or mention these instructions.'
                )
            else:
                request = (
                    f'約1500〜2500字の、独立して読める高品質な日本語の解説文を書いてください。'
                    f'分野：{category}、主題：{topic}、観点：{angle}、多様性キー：{diversity_key}。'
                    '具体的で一般的すぎない小テーマを選び、定義、仕組み、例、限界を含めてください。'
                    '問答形式や指示への言及は避けてください。'
                )
            job = {
                'id': f'ds4p-{i:08d}',
                'category': category,
                'topic': topic,
                'angle': angle,
                'language': language,
                'diversity_key': diversity_key,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': request},
                ],
            }
            f.write(json.dumps(job, ensure_ascii=False) + '\n')
    print(f'jobs={args.count} path={JOBS}')


async def generate(args: argparse.Namespace) -> None:
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit('缺少 httpx：pip install httpx') from exc

    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise SystemExit('未设置 DEEPSEEK_API_KEY')
    if not JOBS.exists():
        raise SystemExit(f'不存在 {JOBS}，先运行 make-jobs')

    completed = {str(x.get('id')) for x in iter_jsonl(RAW) or [] if x.get('id')}
    jobs = [x for x in iter_jsonl(JOBS) or [] if str(x.get('id')) not in completed]
    if args.limit > 0:
        jobs = jobs[:args.limit]
    print(f'total_pending={len(jobs)} completed={len(completed)} concurrency={args.concurrency}')
    if not jobs:
        return

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    write_lock = asyncio.Lock()
    counter_lock = asyncio.Lock()
    done = 0
    failed = 0
    started = time.time()

    timeout = httpx.Timeout(args.timeout, connect=30.0, read=args.timeout, write=60.0, pool=60.0)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(
        base_url='https://api.deepseek.com',
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        timeout=timeout,
        limits=limits,
        http2=True,
    ) as client:
        async def worker(worker_id: int) -> None:
            nonlocal done, failed
            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    return
                last_error: Exception | None = None
                for attempt in range(args.retries):
                    try:
                        payload = {
                            'model': args.model,
                            'messages': job['messages'],
                            'thinking': {'type': 'disabled'},
                            'temperature': args.temperature,
                            'max_tokens': args.max_tokens,
                            'stream': False,
                            'user_id': 'albertlm-distill',
                        }
                        response = await client.post('/chat/completions', json=payload)
                        if response.status_code in {429, 500, 502, 503, 504}:
                            raise RuntimeError(f'HTTP {response.status_code}: {response.text[:300]}')
                        response.raise_for_status()
                        data = response.json()
                        choice = data['choices'][0]
                        content = choice['message'].get('content') or ''
                        if not content.strip():
                            raise RuntimeError('empty output')
                        result = {
                            'id': job['id'],
                            'category': job['category'],
                            'topic': job['topic'],
                            'angle': job['angle'],
                            'language': job['language'],
                            'prompt': job['messages'][-1]['content'],
                            'output': content,
                            'model': data.get('model', args.model),
                            'finish_reason': choice.get('finish_reason'),
                            'usage': data.get('usage', {}),
                            'created': data.get('created'),
                            'request_id': response.headers.get('x-request-id'),
                            'timestamp': time.time(),
                        }
                        async with write_lock:
                            write_jsonl(RAW, result)
                        async with counter_lock:
                            done += 1
                            if done % 10 == 0 or done == len(jobs):
                                elapsed = max(time.time() - started, 1e-6)
                                print(f'done={done}/{len(jobs)} failed={failed} rate={done/elapsed:.2f} req/s', flush=True)
                        last_error = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        if attempt + 1 < args.retries:
                            delay = min(60.0, (2 ** attempt) + random.random() * 2)
                            await asyncio.sleep(delay)
                if last_error is not None:
                    async with write_lock:
                        write_jsonl(ERRORS, {
                            'id': job.get('id'),
                            'error_type': type(last_error).__name__,
                            'error': str(last_error),
                            'timestamp': time.time(),
                        })
                    async with counter_lock:
                        failed += 1
                queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
        await queue.join()
        await asyncio.gather(*workers)
    print(f'finished done={done} failed={failed} raw={RAW}')


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\x00', '')
    text = re.sub(r'\r\n?', '\n', text)
    text = text.replace('\t', '    ')
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


def repeated_line_ratio(text: str) -> float:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) < 6:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def clean(args: argparse.Namespace) -> None:
    if not RAW.exists():
        raise SystemExit(f'不存在 {RAW}')
    seen: set[str] = set()
    kept = 0
    rejected = 0
    chars = 0
    usage_in = 0
    usage_out = 0
    refusal_markers = ('作为一个ai', '作为 ai', 'i cannot comply', 'i can\'t comply', '申し訳ありませんが')
    with CLEAN.open('w', encoding='utf-8') as clean_file, CORPUS.open('w', encoding='utf-8') as corpus_file:
        for item in iter_jsonl(RAW) or []:
            text = normalize_text(str(item.get('output') or ''))
            reason = None
            if item.get('finish_reason') != 'stop':
                reason = 'incomplete'
            elif len(text) < args.min_chars:
                reason = 'too_short'
            elif len(text) > args.max_chars:
                reason = 'too_long'
            elif any(marker in text[:300].lower() for marker in refusal_markers):
                reason = 'refusal'
            elif repeated_line_ratio(text) > args.max_repeated_line_ratio:
                reason = 'repetition'
            digest = hashlib.sha256(re.sub(r'\s+', ' ', text).encode('utf-8')).hexdigest()
            if digest in seen:
                reason = 'duplicate'
            if reason:
                rejected += 1
                continue
            seen.add(digest)
            output = {
                'id': item.get('id'),
                'category': item.get('category'),
                'topic': item.get('topic'),
                'language': item.get('language'),
                'prompt': item.get('prompt'),
                'output': text,
                'sha256': digest,
                'model': item.get('model'),
                'usage': item.get('usage', {}),
            }
            clean_file.write(json.dumps(output, ensure_ascii=False) + '\n')
            corpus_file.write(text + '\n\n')
            kept += 1
            chars += len(text)
            usage = item.get('usage') or {}
            usage_in += int(usage.get('prompt_tokens') or 0)
            usage_out += int(usage.get('completion_tokens') or 0)
    stats = {
        'kept_documents': kept,
        'rejected_documents': rejected,
        'characters': chars,
        'api_prompt_tokens': usage_in,
        'api_completion_tokens': usage_out,
        'clean_jsonl': str(CLEAN),
        'corpus_txt': str(CORPUS),
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def find_special_id(tokenizer, names: list[str]) -> int | None:
    for name in names:
        token_id = tokenizer.token_to_id(name)
        if token_id is not None:
            return int(token_id)
    return None


def tokenize(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit('缺少 numpy/tokenizers') from exc
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise SystemExit(f'不存在 tokenizer: {tokenizer_path}')
    if not CLEAN.exists():
        raise SystemExit(f'不存在 {CLEAN}，先运行 clean')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos_id = find_special_id(tokenizer, ['<bos>', '<s>', '[BOS]'])
    eos_id = find_special_id(tokenizer, ['<eos>', '</s>', '[EOS]'])
    unk_id = tokenizer.token_to_id('<unk>')
    total_tokens = 0
    documents = 0
    with BIN.open('wb') as out:
        for item in iter_jsonl(CLEAN) or []:
            text = str(item['output'])
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if unk_id is not None and unk_id in ids:
                raise RuntimeError(f"document {item.get('id')} contains <unk>")
            if bos_id is not None:
                ids.insert(0, bos_id)
            if eos_id is not None:
                ids.append(eos_id)
            if not ids:
                continue
            np.asarray(ids, dtype=np.uint32).tofile(out)
            total_tokens += len(ids)
            documents += 1
    stats = {}
    if STATS.exists():
        try:
            stats = json.loads(STATS.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            stats = {}
    stats.update({
        'tokenized_documents': documents,
        'albertlm_tokens': total_tokens,
        'binary_path': str(BIN),
        'binary_bytes': BIN.stat().st_size,
        'tokenizer_path': str(tokenizer_path),
        'bos_id': bos_id,
        'eos_id': eos_id,
    })
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def status(_: argparse.Namespace) -> None:
    def count(path: Path) -> int:
        return sum(1 for _ in iter_jsonl(path) or []) if path.exists() else 0
    print(f'jobs={count(JOBS)} raw={count(RAW)} errors={count(ERRORS)} clean={count(CLEAN)}')
    for path in [JOBS, RAW, ERRORS, CLEAN, CORPUS, BIN, STATS]:
        if path.exists():
            print(f'{path}: {path.stat().st_size / 1024 / 1024:.2f} MiB')
    if STATS.exists():
        print(STATS.read_text(encoding='utf-8'))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('make-jobs')
    p.add_argument('--count', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20260715)
    p.set_defaults(func=make_jobs)

    p = sub.add_parser('generate')
    p.add_argument('--model', default='deepseek-v4-pro')
    p.add_argument('--concurrency', type=int, default=32)
    p.add_argument('--max-tokens', type=int, default=1800)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--timeout', type=float, default=900.0)
    p.add_argument('--retries', type=int, default=8)
    p.add_argument('--limit', type=int, default=0, help='0 means all pending jobs')
    p.set_defaults(func=lambda args: asyncio.run(generate(args)))

    p = sub.add_parser('clean')
    p.add_argument('--min-chars', type=int, default=500)
    p.add_argument('--max-chars', type=int, default=30000)
    p.add_argument('--max-repeated-line-ratio', type=float, default=0.35)
    p.set_defaults(func=clean)

    p = sub.add_parser('tokenize')
    p.add_argument('--tokenizer', default=str(ROOT / 'tokenizer' / 'tokenizer.json'))
    p.set_defaults(func=tokenize)

    p = sub.add_parser('status')
    p.set_defaults(func=status)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
