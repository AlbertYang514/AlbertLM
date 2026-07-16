<!-- zh-CN -->

# AlbertLM 数据来源与溯源说明

## 范围

本文档说明 AlbertLM 基座预训练和 tokenizer 构建所使用的数据来源。

训练集本身不会随 AlbertLM 分发。本文档对数据源的列举不代表 AlbertLM 对上游数据重新许可。

## 最终预训练流

最终使用的 packed 训练流约包含 143.57 亿个 AlbertLM tokenizer ID。

其组成包括：

1. 早期构建的中英文 Wikipedia 训练 shard；
2. 下文列出的多语言和代码预训练混合数据；
3. 通过 DeepSeek API 生成的小比例合成数据。

加入合成数据前的 packed 训练流：

- 14,331,990,016 tokens

DeepSeek 合成数据：

- 24,784,909 tokens

最终 packed 训练流：

- 14,356,774,925 tokens

DeepSeek 合成数据约占最终训练流的 0.173%。

## 已审计的多语言和代码混合数据

该部分包含：

- 9,639,077,888 个训练 tokens
- 47,388,672 个验证 tokens

| 来源 | 训练 tokens | 数据集 revision |
|---|---:|---|
| FineWeb-Edu 英文 | 3,000,006,656 | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| FineWeb2 中文 `cmn_Hani` | 2,999,998,464 | `af9c13333eb981300149d5ca60a8e9d659b276b9` |
| FineWeb2 日文 `jpn_Jpan` | 999,999,488 | `af9c13333eb981300149d5ca60a8e9d659b276b9` |
| Wikipedia 英文 | 750,000,000 | 本地 Wikimedia 衍生语料 |
| Wikipedia 中文 | 389,077,735 | 本地 Wikimedia 衍生语料 |
| 宽松许可证代码 | 1,499,998,208 | `17cad72c886a2858e08d4c349a00d6466f54df63` |

混合数据完成时，精确去重索引包含 20,409,541 个文档哈希。

## FineWeb-Edu 和 FineWeb2

使用的数据集：

- `HuggingFaceFW/fineweb-edu`
- `HuggingFaceFW/fineweb-2`

对应 dataset card 将数据库标记为 ODC-By 1.0。

底层网页内容仍可能受原始发布者权利约束，Common Crawl 的使用条款也可能适用。AlbertLM 不重新分发这些数据集或其处理后的正文。

## Wikipedia 与 Wikimedia

中英文 Wikipedia 衍生文本用于：

- tokenizer 训练；
- 早期 Wikipedia 训练 shard；
- 已审计的多语言预训练混合数据。

Wikipedia 文本主要依据 CC BY-SA 4.0，并在适用情况下同时依据 GNU Free Documentation License 提供。

AlbertLM 不重新分发 Wikipedia 文章正文。

若能从旧构建记录中恢复，首次发布正式权重前应补充早期 Wikipedia 语料的准确 dump 日期。

Wikimedia Foundation 不为 AlbertLM 提供背书。

## 代码数据

实际使用的数据集：

- `bigcode/the-stack-dedup`
- revision：`17cad72c886a2858e08d4c349a00d6466f54df63`

仅接受通过项目宽松许可证筛选策略的记录。

修复并验证后的 provenance 包含：

- 649,151 条记录；
- 649,151 个唯一 `(source, hash_xxh3_64)`；
- 源仓库名称；
- 源 commit；
- 编程语言；
- 文件扩展名；
- 自动检测的许可证信息。

代码正文不会随 AlbertLM 重新分发。

当前完整许可证扫描以 MIT、Apache-2.0、BSD 系列、Unlicense、CC0、ISC、Zlib 等宽松许可证为主。

扫描中未发现 GPL、AGPL、LGPL、SSPL、BUSL、NonCommercial、NoDerivatives、missing 或 unknown 的直接命中。`CNRI-Python-GPL-Compatible` 是许可证名称中的兼容性描述，不等同于引入 GPL 许可证。

自动许可证检测并非绝对准确。模型仍可能生成与公开源代码相似的内容，因此生成代码在实际使用前应独立进行安全和许可证审查。

## DeepSeek 合成数据

合成文档通过 DeepSeek API 生成，模型标识为：

- `deepseek-v4-pro`

审计统计：

- 请求任务：10,000
- 保留文档：9,869
- 拒绝文档：131
- API prompt tokens：1,841,871
- API completion tokens：19,151,185
- AlbertLM tokenizer IDs：24,784,909

适用的 DeepSeek 服务条款明确允许将输入和输出用于衍生产品开发及训练其他模型，并明确举例包括模型蒸馏。

原始 prompt、API response、清洗后的 JSONL 和合成语料正文不会公开分发。

DeepSeek 不为 AlbertLM 提供背书。

## 数据分发政策

以下内容不属于公开发布物：

- 原始网页文本；
- 处理后的训练正文；
- DeepSeek prompt 与 response JSONL；
- tokenized `.bin` 数据；
- Hugging Face cache；
- 精确去重工作文件；
- 私有 API 调用记录；
- 训练日志。

AlbertLM 公开的是：

- 源代码；
- tokenizer；
- 模型配置；
- 模型权重；
- 评测结果；
- 数据来源说明；
- 必要的第三方声明。

## 已知限制

训练数据中仍可能存在：

- 错误或过时信息；
- 受版权保护的公开网页内容；
- 公开来源中的个人信息；
- 不良语言和社会偏见；
- 未被上游过滤移除的代码密钥；
- 重复或近似重复文档；
- 自动许可证识别错误。

在发布模型权重前，应运行记忆与敏感字符串评测，包括：

- API key 模式；
- 私钥头部标记；
- 邮箱地址；
- 密码或 credential 模式；
- 异常长的逐字续写；
- 与训练材料高度相似的代码片段。

---

<!-- en -->

# AlbertLM Data Provenance

## Scope

This document describes the datasets used for AlbertLM base pretraining
and tokenizer construction.

The training datasets themselves are not distributed with AlbertLM.
Their inclusion here does not relicense upstream content.

## Final pretraining stream

The active packed training stream contains approximately 14.357 billion
AlbertLM tokenizer IDs.

The final stream consists of:

1. a legacy English and Chinese Wikipedia-derived training shard;
2. the audited multilingual and code pretraining mixture described below;
3. a small synthetic component generated through the DeepSeek API.

The packed stream before synthetic-data injection contains
14,331,990,016 token IDs.

The DeepSeek synthetic component adds 24,784,909 token IDs.

The resulting packed stream contains 14,356,774,925 token IDs.

## Audited multilingual and code mixture

The audited pretraining mixture contains 9,639,077,888 training tokens
and 47,388,672 validation tokens.

| Source | Train tokens | Dataset revision |
|---|---:|---|
| FineWeb-Edu English | 3,000,006,656 | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| FineWeb2 Chinese (`cmn_Hani`) | 2,999,998,464 | `af9c13333eb981300149d5ca60a8e9d659b276b9` |
| FineWeb2 Japanese (`jpn_Jpan`) | 999,999,488 | `af9c13333eb981300149d5ca60a8e9d659b276b9` |
| Wikipedia English | 750,000,000 | Local Wikimedia-derived corpus |
| Wikipedia Chinese | 389,077,735 | Local Wikimedia-derived corpus |
| Permissive code | 1,499,998,208 | `17cad72c886a2858e08d4c349a00d6466f54df63` |

The exact-deduplication index contained 20,409,541 document hashes at
mixture completion.

## FineWeb-Edu and FineWeb2

Repositories:

- `HuggingFaceFW/fineweb-edu`
- `HuggingFaceFW/fineweb-2`

The dataset cards identify these databases under ODC-By 1.0.
Underlying web content may remain subject to rights held by the original
publishers, and Common Crawl terms may also apply.

AlbertLM does not redistribute these datasets or their processed text.

## Wikipedia and Wikimedia

English and Chinese Wikipedia-derived text was used for:

- tokenizer training;
- the legacy Wikipedia training shard;
- the audited multilingual mixture.

Wikipedia text is provided by Wikimedia under CC BY-SA 4.0 and, where
applicable, the GNU Free Documentation License.

AlbertLM does not redistribute Wikipedia article text.

The exact dump dates for the earliest legacy tokenizer corpus should be
added before the first public weight release if they can be reconstructed
from local build records.

Wikimedia Foundation does not endorse AlbertLM.

## Code data

Repository:

- `bigcode/the-stack-dedup`
- revision: `17cad72c886a2858e08d4c349a00d6466f54df63`

Only records passing the project's permissive-license selection policy
were accepted.

The final provenance journal contains:

- 649,151 records;
- 649,151 unique `(source, hash_xxh3_64)` keys;
- repository name, commit, language, extension, and detected license
  metadata where available.

The code corpus itself is not redistributed.

The recorded licenses are predominantly MIT, Apache-2.0, BSD-family,
Unlicense, CC0, ISC, Zlib, and other permissive licenses. Source-specific
license and attribution obligations remain applicable.

Automated license detection is not infallible. The model may still
produce code similar to public source material, and generated code should
be independently reviewed before use.

## DeepSeek synthetic data

Synthetic documents were generated through the DeepSeek API using the
model identifier `deepseek-v4-pro`.

Audited statistics:

- requested jobs: 10,000
- retained documents: 9,869
- rejected documents: 131
- API prompt tokens: 1,841,871
- API completion tokens: 19,151,185
- AlbertLM tokenizer IDs: 24,784,909

The applicable DeepSeek terms explicitly permit use of inputs and outputs
for derivative-product development and training other models, including
model distillation.

The raw prompts, API responses, cleaned JSONL, and generated corpus are
not distributed.

DeepSeek does not endorse AlbertLM.

## Data distribution policy

The following files are not part of the public release:

- raw crawled text;
- processed training text;
- JSONL synthetic generations;
- tokenized `.bin` datasets;
- Hugging Face cache files;
- exact-dedup working state;
- private API records;
- training logs.

AlbertLM releases source code, tokenizer assets, model metadata, model
weights, evaluation results, and sufficiently detailed provenance
documentation instead.

## Known limitations

The training corpus may contain:

- incorrect or outdated information;
- copyrighted public web content;
- personal information present in public sources;
- undesirable language or social bias;
- source-code secrets that escaped upstream filtering;
- duplicated or near-duplicated documents;
- inaccurate automatically detected license metadata.

Before weight release, the project should run memorization and sensitive
string evaluations, including checks for API keys, private-key markers,
email addresses, and unusually long verbatim continuations.
