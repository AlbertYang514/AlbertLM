<!-- zh-CN -->

# AlbertLM-1.7B-Base 模型卡

## 当前状态

模型仍在训练中，公开权重尚未发布。

本文档是计划发布的 `AlbertLM-1.7B-Base` 模型卡草案。正式发布时将补充完整评测结果、权重哈希和可复现加载说明。

## 模型信息

- 模型类型：Decoder-only Transformer
- 参数量：约 1.678B
- 层数：28
- 隐藏维度：2,048
- FFN 中间维度：6,144
- 注意力头：16
- KV 头：8
- 词表大小：65,536
- 上下文长度：2,048
- 激活函数：SwiGLU
- 归一化：RMSNorm
- 位置编码：RoPE
- 训练精度：BF16 与 FP8
- 主要语言：中文、英文、日文
- 附加领域：宽松许可证源代码

## 预期用途

AlbertLM-1.7B-Base 主要用于：

- 语言模型研究；
- 预训练与缩放实验；
- tokenizer 与模型架构研究；
- 微调和对齐实验；
- 单卡模型训练的教学与工程研究。

它是 base model，不应预期它能稳定遵循自然语言指令。

## 不建议用途

不要直接将模型视为以下领域的可靠权威：

- 医疗决策；
- 法律意见；
- 金融决策；
- 安全关键控制；
- 身份认证或安全判断；
- 无人工审查的生产代码生成。

## 训练数据

模型使用的训练数据包括：

- Wikipedia 衍生文本；
- FineWeb-Edu；
- FineWeb2；
- 从 The Stack Dedup 中筛选的宽松许可证代码；
- 通过 DeepSeek API 生成的小比例合成数据。

训练数据本身不会公开分发。

准确数据 revision 和审计统计见 `DATA_PROVENANCE.md`。

## 评测计划

预训练完成后应至少公开：

- validation loss 和 perplexity；
- 中文、英文、日文分域 loss；
- 代码分域 loss；
- 固定 prompt 的确定性生成结果；
- 多随机种子采样结果；
- 重复、语言切换和主题保持指标；
- base-model likelihood benchmark；
- 记忆与敏感字符串检查。

## 已知限制

模型可能生成：

- 虚构或错误信息；
- 带有偏见、不安全或令人不适的内容；
- 重复、语言混杂或失去连贯性的文本；
- 不安全或许可证存在问题的代码；
- 与公开训练材料相似的片段。

模型尚未经过指令微调、偏好对齐或完整安全训练。

## 许可证

计划发布的模型权重使用 Apache License 2.0。

训练数据仍分别受其上游许可证和服务条款约束，不包含在模型权重许可证内。

---

<!-- en -->

# AlbertLM-1.7B-Base Model Card

## Status

Training is in progress. Public model weights have not yet been released.

This file is the working model card for the planned
`AlbertLM-1.7B-Base` release.

## Model details

- Model type: decoder-only Transformer
- Parameters: approximately 1.678B
- Layers: 28
- Hidden size: 2,048
- Intermediate size: 6,144
- Attention heads: 16
- KV heads: 8
- Vocabulary size: 65,536
- Context length: 2,048
- Activation: SwiGLU
- Normalization: RMSNorm
- Position encoding: RoPE
- Training precision: BF16 and FP8
- Primary languages: Chinese, English, Japanese
- Additional domain: permissively licensed source code

## Intended use

AlbertLM-1.7B-Base is intended for:

- language-model research;
- pretraining and scaling experiments;
- tokenizer and architecture research;
- fine-tuning and alignment experiments;
- educational study of single-GPU model training.

It is a base model and is not expected to reliably follow instructions.

## Out-of-scope use

The model should not be treated as a reliable authority for:

- medical decisions;
- legal advice;
- financial decisions;
- safety-critical control;
- authentication or security decisions;
- unsupervised production code generation.

## Training data

The model was trained using a mixture of Wikipedia-derived text,
FineWeb-Edu, FineWeb2, permissively selected code from The Stack Dedup,
and a small synthetic component generated through the DeepSeek API.

Training data is not distributed.

See `DATA_PROVENANCE.md` for exact revisions and audited statistics.

## Evaluation

Evaluation results will be added after base pretraining is complete.

The release evaluation should include:

- validation loss and perplexity;
- language-specific loss;
- code-specific loss;
- fixed-prompt deterministic generation;
- multi-seed sampled generation;
- repetition and language-switching metrics;
- base-model likelihood benchmarks;
- memorization and sensitive-string checks.

## Limitations

The model may generate fabricated, biased, unsafe, repetitive, or
copyright-sensitive content.

It may reproduce fragments similar to public training material and may
generate insecure or incorrectly licensed code.

The model has not been instruction-tuned or preference-aligned.

## License

The planned model-weight license is Apache License 2.0.

Training datasets remain governed by their respective licenses and
terms and are not covered by the model license.
