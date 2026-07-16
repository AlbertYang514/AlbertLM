<!-- zh-CN -->

# AlbertLM

AlbertLM 是一个从零开始训练的 decoder-only 语言模型研究项目，目标是在单张消费级 GPU 上完成 tokenizer、数据构建、Dense 预训练、FP8 训练、运行时评测，以及后续指令微调和混合专家实验。

## 当前状态

- 基座模型预训练：进行中
- 源代码：已公开
- tokenizer：已公开
- 模型权重：尚未发布
- 训练数据：不公开分发
- 首个计划发布的权重：`AlbertLM-1.7B-Base`

基座模型完成预训练后，将经过评测、Safetensors 导出、权重一致性验证和敏感信息检查，再发布正式权重。

## 模型架构

| 项目 | 数值 |
|---|---:|
| 架构 | Decoder-only Transformer |
| 参数量 | 约 1.678B |
| 层数 | 28 |
| 隐藏维度 | 2,048 |
| FFN 中间维度 | 6,144 |
| 注意力头 | 16 |
| KV 头 | 8 |
| 词表大小 | 65,536 |
| 上下文长度 | 2,048 |
| 激活函数 | SwiGLU |
| 归一化 | RMSNorm |
| 位置编码 | RoPE |
| 训练精度 | BF16 / FP8 |

## 仓库结构

- `albertlm/`：模型实现
- `train/`：预训练与运行时评测
- `configs/`：模型与 DeepSpeed 配置
- `scripts/`：数据、tokenizer、训练、监控和控制脚本
- `tests/`：回归测试和运行时测试
- `tokenizer/`：AlbertLM tokenizer 资产
- `docs/`：运维与发布文档
- `metadata/`：机器可读的公开元数据

模型 checkpoint、优化器状态、训练集、缓存、日志和私有审计材料不会进入普通 Git 历史。

## 训练数据

AlbertLM 不重新分发训练语料。

基座预训练流主要包含：

- FineWeb-Edu 英文教育网页文本
- FineWeb2 中文和日文网页文本
- 中英文 Wikipedia 衍生文本
- 从 The Stack Dedup 中筛选的宽松许可证代码
- 通过 DeepSeek API 生成的小比例合成数据

准确的数据集 revision、token 数、许可证和已知限制见：

- [DATA_PROVENANCE.md](DATA_PROVENANCE.md)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Tokenizer

AlbertLM 使用词表大小为 65,536 的 Byte-Level BPE tokenizer，主要通过中英文 Wikipedia 衍生文本训练，并覆盖日文和代码。

仓库中的 tokenizer 文件采用 Apache License 2.0。用于训练 tokenizer 的原始 Wikimedia 文本仍受其原始许可证约束，且不会随本项目重新分发。

## 许可证

除特别说明外，本仓库中的源代码、配置、文档和 tokenizer 资产使用 Apache License 2.0。

训练数据不包含在仓库中，也不会被 AlbertLM 重新许可。

未来发布的模型权重将附带独立许可证声明，当前计划使用 Apache License 2.0。

相关文件：

- [LICENSE](LICENSE)
- [NOTICE](NOTICE)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## 已知限制

AlbertLM 当前是研究性质的 base model，并非经过指令微调的对话助手。

它可能：

- 生成错误或虚构的信息；
- 继承训练数据中的偏见和不良内容；
- 生成不安全或许可证不明确的代码；
- 复现与公开训练材料相似的片段；
- 在长文本中发生重复、语言切换或主题漂移。

不要将未发布或中间 checkpoint 用于医疗、法律、金融、安全控制或其他高风险决策。

## 引用

引用信息见 [CITATION.cff](CITATION.cff)。

---

<!-- en -->

# AlbertLM

AlbertLM is a from-scratch decoder-only language-model research project
developed on a single consumer GPU.

The project covers tokenizer training, multilingual and code corpus
construction, dense pretraining, FP8 training, runtime evaluation,
checkpoint management, and future instruction tuning and mixture-of-experts
experiments.

## Project status

- Base model pretraining: in progress
- Public source code: available
- Public model weights: not released yet
- Training data: not distributed
- Planned first weight release: AlbertLM-1.7B-Base

The first public weight release will occur after base pretraining,
evaluation, export to model-only Safetensors, and release validation.

## Model architecture

| Property | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Parameters | Approximately 1.678B |
| Layers | 28 |
| Hidden size | 2,048 |
| Intermediate size | 6,144 |
| Attention heads | 16 |
| KV heads | 8 |
| Vocabulary size | 65,536 |
| Context length | 2,048 |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| Training precision | BF16 / FP8 |

## Repository contents

- `albertlm/`: model implementation
- `train/`: pretraining and runtime evaluation
- `configs/`: model and DeepSpeed configurations
- `scripts/`: data, tokenizer, training, monitoring, and control scripts
- `tests/`: regression and runtime tests
- `tokenizer/`: AlbertLM tokenizer assets
- `docs/`: operational and release documentation
- `metadata/`: machine-readable public metadata

Model checkpoints, optimizer states, training datasets, caches, logs,
and private audit artifacts are intentionally excluded.

## Training data

AlbertLM does not redistribute its training corpus.

The base pretraining stream includes:

- English educational web text from FineWeb-Edu
- Chinese and Japanese web text from FineWeb2
- English and Chinese Wikipedia-derived text
- permissively licensed code selected from The Stack Dedup
- a small synthetic component generated through the DeepSeek API

Exact dataset revisions, token allocations, terms, and known limitations
are documented in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## Tokenizer

AlbertLM uses a 65,536-token Byte-Level BPE tokenizer trained primarily
on English and Chinese Wikipedia-derived text, with multilingual and code
coverage.

The tokenizer files distributed in this repository are licensed under
Apache License 2.0. The source text used to train the tokenizer remains
subject to its original Wikimedia licenses and is not redistributed.

## License

Unless otherwise stated, source code, configuration files, documentation,
and tokenizer assets in this repository are licensed under the
Apache License 2.0.

Training datasets are not included and are not relicensed by this project.

Released model weights will include their own license statement. The
planned license for AlbertLM base weights is Apache License 2.0.

See:

- [LICENSE](LICENSE)
- [NOTICE](NOTICE)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Limitations

AlbertLM is a research base model, not an instruction-following assistant.

It may:

- generate incorrect or fabricated information;
- reproduce biases or undesirable content from training sources;
- generate code with security or licensing problems;
- reproduce fragments similar to public training material;
- switch languages or lose coherence over long generations.

Do not use unreleased or intermediate checkpoints in safety-critical,
medical, legal, financial, or production decision-making systems.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).
