<!-- zh-CN -->

# 第三方声明

AlbertLM 在 tokenizer 构建和模型训练过程中使用了第三方项目、数据集、软件和服务。

本文档仅用于来源识别和归属说明，不代表 AlbertLM 对任何上游数据、软件或内容重新许可。

## Wikimedia 项目

中英文 Wikipedia 衍生文本用于 tokenizer 构建和基座预训练。

Wikipedia 文本主要依据 Creative Commons Attribution-ShareAlike 4.0，并在适用情况下同时依据 GNU Free Documentation License 提供。

AlbertLM 不重新分发 Wikipedia 文章正文。

Wikimedia Foundation 不为 AlbertLM 提供背书。

## FineWeb-Edu 与 FineWeb2

AlbertLM 使用了：

- `HuggingFaceFW/fineweb-edu`
- `HuggingFaceFW/fineweb-2`

相应 dataset card 将数据库标记为 ODC-By 1.0。底层网页文档和 Common Crawl 材料仍可能受其他权利与条款约束。

这些数据集不会随 AlbertLM 重新分发。

## The Stack Dedup

AlbertLM 使用了经过宽松许可证筛选的：

- `bigcode/the-stack-dedup`

源代码仓库仍保留其原始许可证、归属要求、NOTICE 和版权归属。

AlbertLM 保存逐条 provenance 元数据，但不重新分发源代码训练语料。

## DeepSeek

AlbertLM 的一小部分合成训练数据通过 DeepSeek API 生成。

适用的服务条款允许将输入和输出用于衍生产品开发和训练其他模型，包括模型蒸馏。

原始 API 输出不会公开分发。

DeepSeek 不为 AlbertLM 提供背书。

## 软件依赖

AlbertLM 使用的第三方软件包括但不限于：

- PyTorch
- DeepSpeed
- Hugging Face libraries
- NVIDIA Transformer Engine
- NumPy
- xxHash
- tokenizers

这些依赖仍受各自许可证约束，不会因 AlbertLM 使用 Apache License 2.0 而被重新许可。

---

<!-- en -->

# Third-Party Notices

AlbertLM uses or was trained with material obtained from third-party
projects and services.

This file provides attribution and identification only. It does not
relicense upstream datasets, software, or content.

## Wikimedia projects

English and Chinese Wikipedia-derived text was used for tokenizer
construction and base pretraining.

Wikipedia text is available under Creative Commons Attribution-ShareAlike
4.0 and, where applicable, the GNU Free Documentation License.

Wikimedia Foundation does not endorse AlbertLM.

## FineWeb-Edu and FineWeb2

AlbertLM used data from:

- HuggingFaceFW/fineweb-edu
- HuggingFaceFW/fineweb-2

The corresponding dataset cards identify ODC-By 1.0 for the database.
Underlying web documents and Common Crawl material may remain subject to
additional rights and terms.

The datasets are not redistributed by AlbertLM.

## The Stack Dedup

AlbertLM used a permissive-license-filtered subset of:

- bigcode/the-stack-dedup

Individual source repositories retain their original licenses,
attribution requirements, notices, and copyright ownership.

AlbertLM preserves per-record provenance metadata but does not
redistribute the source-code corpus.

## DeepSeek

A small synthetic training component was generated using the DeepSeek API.

The applicable service terms permit using inputs and outputs to develop
derivative products and train other models, including through model
distillation.

Raw API outputs are not distributed.

DeepSeek does not endorse AlbertLM.

## Software dependencies

AlbertLM depends on third-party software including PyTorch, DeepSpeed,
Hugging Face libraries, NVIDIA Transformer Engine, NumPy, xxHash, and
other packages.

Those dependencies remain governed by their respective licenses.
They are not relicensed by the AlbertLM Apache License 2.0 notice.
