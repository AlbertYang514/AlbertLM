<!-- zh-CN -->

# 参与 AlbertLM 开发

欢迎通过 GitHub issue 和 pull request 参与 AlbertLM。

## 基本要求

- 修改应保持聚焦、可审查；
- 行为变化应增加或更新测试；
- 修改 Python 文件后应运行语法检查和相关单元测试；
- 不得提交训练数据、checkpoint、优化器状态、缓存、日志、密钥或私有审计材料；
- 不得将生成的模型权重直接提交到普通 Git 历史；
- 应保留第三方归属与许可证信息；
- 影响可复现性或数据来源的修改必须同步更新文档。

## 基本检查

至少运行：

    python -m py_compile <修改过的 Python 文件>
    python -m unittest -v
    git diff --check

涉及安全、checkpoint、FP8 或数据流水线的修改，应增加针对性的回归测试。

---

<!-- en -->

# Contributing to AlbertLM

Contributions are welcome through GitHub issues and pull requests.

## Requirements

- Keep changes focused and reviewable.
- Add or update tests for behavioral changes.
- Run relevant Python compilation and unit tests.
- Do not commit training datasets, model checkpoints, optimizer states,
  caches, logs, credentials, or private audit artifacts.
- Do not add generated model weights to ordinary Git history.
- Preserve attribution and third-party license information.
- Document changes that affect reproducibility or data provenance.

## Development checks

At minimum:

```bash
python -m py_compile <changed Python files>
python -m unittest -v
git diff --check
Security-sensitive, checkpoint, FP8, and data-pipeline changes should
include a focused regression test.
