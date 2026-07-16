<!-- zh-CN -->

# AlbertLM 模型权重发布检查表

## Checkpoint 选择

- [ ] 选择已经完成验证的干净 checkpoint
- [ ] 永远不得使用已知受污染的 step 2519 checkpoint
- [ ] 记录 optimizer step
- [ ] 记录已训练 token 数
- [ ] 记录对应的 Git commit
- [ ] 私下保存原始训练 checkpoint
- [ ] 确认 checkpoint 没有经过异常梯度更新

## 模型导出

- [ ] 只导出模型权重
- [ ] 不发布 optimizer state
- [ ] 不发布 scheduler state
- [ ] 不发布 RNG state
- [ ] 不发布 DeepSpeed 内部状态
- [ ] 转换为 Safetensors
- [ ] 按合理大小拆分权重文件
- [ ] 生成 `config.json`
- [ ] 生成 `generation_config.json`
- [ ] 包含完整 tokenizer 文件
- [ ] 必要时生成 Safetensors index
- [ ] 为全部发布文件计算 SHA-256

## 权重一致性验证

- [ ] 在干净环境中加载导出模型
- [ ] 使用同一输入比较原 checkpoint 与 Safetensors logits
- [ ] 设置明确的数值误差门槛
- [ ] 确认全部参数为有限值
- [ ] 确认全部 logits 为有限值
- [ ] 运行固定 prompt 的确定性生成
- [ ] 运行多随机种子采样生成
- [ ] 验证 tokenizer ID 完全一致
- [ ] 验证 EOS、BOS、PAD 和特殊 token 配置

## 能力评测

- [ ] validation loss
- [ ] perplexity
- [ ] 中文分域 loss
- [ ] 英文分域 loss
- [ ] 日文分域 loss
- [ ] 代码分域 loss
- [ ] 固定 prompt 生成
- [ ] 多随机种子生成
- [ ] 重复率
- [ ] 语言切换率
- [ ] 主题保持
- [ ] base-model likelihood benchmark
- [ ] 代码语法或可解析性检查

## 安全与记忆检查

- [ ] 检查 API key 模式
- [ ] 检查私钥头部标记
- [ ] 检查邮箱地址
- [ ] 检查密码或 credential 模式
- [ ] 检查异常长逐字续写
- [ ] 检查高度相似的代码片段
- [ ] 检查个人信息模式
- [ ] 记录并公开已知限制

## 发布内容

- [ ] 模型 Safetensors
- [ ] `config.json`
- [ ] `generation_config.json`
- [ ] tokenizer 文件
- [ ] 模型卡
- [ ] Apache-2.0 LICENSE
- [ ] NOTICE
- [ ] DATA_PROVENANCE
- [ ] THIRD_PARTY_NOTICES
- [ ] 权重 SHA-256
- [ ] 评测结果
- [ ] 加载示例
- [ ] 推理依赖版本
- [ ] 对应 Git commit

## 禁止发布的内容

- [ ] 不包含原始训练数据
- [ ] 不包含处理后的训练数据
- [ ] 不包含 DeepSeek prompt 或 response JSONL
- [ ] 不包含 optimizer state
- [ ] 不包含 scheduler state
- [ ] 不包含 pickle 训练 checkpoint
- [ ] 不包含 API key
- [ ] 不包含私人绝对路径
- [ ] 不包含训练日志
- [ ] 不包含 Hugging Face cache
- [ ] 不包含私有审计材料

## 发布前最终确认

- [ ] 权重许可证和仓库许可证表述一致
- [ ] 数据来源与实际训练 revision 一致
- [ ] 模型卡明确说明这是 base model
- [ ] 模型卡明确说明未经指令对齐
- [ ] 模型卡明确说明训练数据不分发
- [ ] 模型卡明确说明 DeepSeek、Wikimedia、Hugging Face 和 BigCode 不提供背书
- [ ] Git 仓库不包含权重和训练数据
- [ ] 模型仓库不包含训练状态
- [ ] 从全新环境完成一次端到端加载测试

---

<!-- en -->

# AlbertLM Model Weight Release Checklist

## Checkpoint selection

- [ ] Select a clean and validated checkpoint
- [ ] Never use the known polluted step 2519 checkpoint
- [ ] Record the optimizer step
- [ ] Record the trained-token count
- [ ] Record the corresponding Git commit
- [ ] Preserve the original training checkpoint privately
- [ ] Confirm that the checkpoint did not follow an anomalous gradient update

## Model export

- [ ] Export model weights only
- [ ] Do not publish optimizer state
- [ ] Do not publish scheduler state
- [ ] Do not publish RNG state
- [ ] Do not publish DeepSpeed internal state
- [ ] Convert weights to Safetensors
- [ ] Split weights into practical shard sizes
- [ ] Generate `config.json`
- [ ] Generate `generation_config.json`
- [ ] Include all tokenizer files
- [ ] Generate a Safetensors index where required
- [ ] Compute SHA-256 checksums for every release file

## Weight validation

- [ ] Load the exported model in a clean environment
- [ ] Compare source-checkpoint and Safetensors logits using identical input
- [ ] Define an explicit numerical tolerance
- [ ] Confirm that every parameter is finite
- [ ] Confirm that every logit is finite
- [ ] Run deterministic fixed-prompt generation
- [ ] Run sampled generation with multiple seeds
- [ ] Confirm identical tokenizer IDs
- [ ] Validate EOS, BOS, PAD, and special-token settings

## Capability evaluation

- [ ] Validation loss
- [ ] Perplexity
- [ ] Chinese-domain loss
- [ ] English-domain loss
- [ ] Japanese-domain loss
- [ ] Code-domain loss
- [ ] Fixed-prompt generation
- [ ] Multi-seed sampled generation
- [ ] Repetition rate
- [ ] Language-switching rate
- [ ] Topic retention
- [ ] Base-model likelihood benchmarks
- [ ] Code syntax or parser checks

## Safety and memorization

- [ ] Probe for API-key patterns
- [ ] Probe for private-key markers
- [ ] Probe for email addresses
- [ ] Probe for password or credential patterns
- [ ] Test unusually long verbatim continuation
- [ ] Test highly similar code continuation
- [ ] Probe for personal-information patterns
- [ ] Document known limitations

## Release contents

- [ ] Model Safetensors
- [ ] `config.json`
- [ ] `generation_config.json`
- [ ] Tokenizer files
- [ ] Model card
- [ ] Apache-2.0 LICENSE
- [ ] NOTICE
- [ ] DATA_PROVENANCE
- [ ] THIRD_PARTY_NOTICES
- [ ] Weight SHA-256 checksums
- [ ] Evaluation results
- [ ] Loading example
- [ ] Inference dependency versions
- [ ] Corresponding Git commit

## Excluded material

- [ ] No raw training data
- [ ] No processed training data
- [ ] No DeepSeek prompt or response JSONL
- [ ] No optimizer state
- [ ] No scheduler state
- [ ] No pickle-based training checkpoint
- [ ] No API keys
- [ ] No private absolute filesystem paths
- [ ] No training logs
- [ ] No Hugging Face cache
- [ ] No private audit artifacts

## Final release confirmation

- [ ] Weight-license wording matches repository-license wording
- [ ] Data provenance matches the actual training revisions
- [ ] The model card clearly identifies the release as a base model
- [ ] The model card states that the model is not instruction-aligned
- [ ] The model card states that training data is not distributed
- [ ] The model card states that DeepSeek, Wikimedia, Hugging Face, and BigCode do not endorse the project
- [ ] The Git repository contains no weights or training data
- [ ] The model repository contains no training state
- [ ] End-to-end loading succeeds in a fresh environment
