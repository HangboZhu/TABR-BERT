# Plan: 将 TABR-BERT 迁移到 ESM2 进行 TCR-pMHC 预测

## 背景

当前 TABR-BERT 使用自定义的 4 层 BERT 编码器（d_model=256）来提取 TCR 和 pMHC 的 embedding。我们需要将两者都替换为预训练的 ESM2-650M 模型（`esm2_t33_650M_UR50D`，hidden_size=1280，33 层）。分类头的架构（4 个并行投影 + MLP）保持不变，但维度需要适配。预训练脚本删除，因为 ESM2 已经是预训练好的模型。

## 环境

- UV 环境: `/disk1/zhuhb/project/2026_tcr_llm/`（Python 3.13.2, torch 2.10.0, transformers 4.57.6, swanlab 0.7.8）
- ESM2 模型: `/disk1/zhuhb/project/2026_tcr_llm/pretrain_model/esm2_t33_650M_UR50D/`（HuggingFace 格式）
- GPU: CUDA 设备 7（单卡，不使用 DataParallel）
- 运行命令: `cd /disk1/zhuhb/project/2026_tcr_llm && uv run ...`

## 第 1 步：删除不需要的文件

删除 5 个不再使用的文件：
- `TABR-BERT/pre_train_tcr_embedding_model.py`
- `TABR-BERT/pre_train_pmhc_embedding_model.py`
- `TABR-BERT/bert_tcr.py`
- `TABR-BERT/bert_pmhc.py`
- `TABR-BERT/embeding.py`

## 第 2 步：在 `utils.py` 中添加 ESM2 embedding 提取函数

添加共享函数 `extract_esm2_embeddings(sequences, tokenizer, model, device, max_length, batch_size)`：
- 使用 `EsmTokenizer` 进行 tokenization（自动添加 BOS/EOS，pad 到 `max_length + 2`，截断）
- 运行 ESM2 推理（frozen，float16，`torch.no_grad()`）
- 从输出中去掉 BOS/EOS：`output[:, 1:-1, :]` → 形状 `(batch, max_length, 1280)`
- 展平为：`(batch, max_length * 1280)` 用于缓存
- 分批处理，拼接所有结果

## 第 3 步：重写 `train_tcr_pmhc_prediction_model.py`

### 3a. 替换 import
- 删除：`from embeding import *`、`from bert_pmhc import BERT as pmhc_net`、`from bert_tcr import BERT as tcr_net`
- 添加：`from transformers import EsmTokenizer, EsmModel`、`import swanlab`
- 从 `utils` 导入 `extract_esm2_embeddings`

### 3b. 更新 argparse 参数
- 删除：`--tcr_model`、`--pmhc_model`、`--GPUs`
- 添加：`--esm2_model_dir`（默认值：ESM2 路径）、`--gpu`（默认值：7）、`--swanlab_project`、`--swanlab_experiment`
- 更新：`--pmhc_d_model` 和 `--tcr_d_model` 默认值改为 1280

### 3c. 设备设置
- 将 DataParallel 多卡逻辑替换为：`device = torch.device(f"cuda:{args.gpu}")`
- 所有 `.cuda()` 调用 → `.to(device)`

### 3d. 删除旧的 tokenization 函数
- 删除 `aa_to_index()`、`tcr_make_data()`、`pmhc_make_data()` — 不再需要（ESM2 tokenizer 负责处理）

### 3e. 替换 embedding 提取代码块（原 ~177-226 行）
- 加载一个共享的 ESM2 模型（float16，frozen，`.eval()`）
- 构建 pMHC 序列：直接字符串拼接 `allele_伪序列 + peptide`（无分隔符）
- 提取 TCR embedding：CDR3 字符串 → ESM2 → `(N, 30, 1280)` → 缓存
- 提取 pMHC embedding：拼接字符串 → ESM2 → `(N, 54, 1280)` → 缓存
- 提取健康 TCR embedding：同 TCR 提取方式
- 提取完成后释放 ESM2 的 GPU 显存

### 3f. 更新分类器 `tcr_pmhc` 类
保持 4 个并行投影的架构模式，适配维度：

| 组件 | 旧维度 | 新维度 |
|------|--------|--------|
| d_model | 256 | 1280 |
| tcr_linear | Linear(256,1) | Linear(1280,1) |
| pmhc_linear | Linear(256,1) | Linear(1280,1) |
| _tcr_linear | Linear(30,1) | Linear(30,1) — 不变 |
| _pmhc_linear | Linear(54,1) | Linear(54,1) — 不变 |
| 拼接维度 | 30+54+256+256=596 | 30+54+1280+1280=2644 |
| Dense MLP | 596→200→100→50→1 | 2644→512→256→64→1 |

`forward()` 逻辑完全不变 — 仅维度变化。

### 3g. 移除 DataParallel 包装
- `model = nn.DataParallel(model, ...)` → `model.to(device)`
- Loss 函数和优化器保持不变

### 3h. 添加 SwanLab 日志记录
- 训练循环前调用 `swanlab.init()`，传入配置（lr、batch_size、esm2_model、embedding_dim 等）
- 每 200 个训练 step 记录：`train/loss`、`train/accuracy`
- 每 50 个验证 step 记录：`val/loss`、`val/accuracy`
- 每 epoch 结束记录：`train/epoch_loss`、`val/epoch_loss`、`lr`、`epoch`
- 训练结束后调用 `swanlab.finish()`

### 3i. 保持不变的部分
- 数据加载逻辑（CSV 读取、通过 mhcnames 规范化 allele 名、伪序列映射）
- `TCR_PMHC_loss` 函数（基于 margin 的对比损失）
- `get_data()` 函数结构（训练/验证划分、负采样）
- 训练循环结构（AdamW、ReduceLROnPlateau、EarlyStopping、每 5 个 epoch 重新生成负样本）
- 从健康 TCR 中进行负采样

## 第 4 步：更新 `predict_tcr_pmhc_binding.py`

与第 3 步做相同的镜像更改，但是用于推理：
- 替换 import、argparse、设备设置
- 相同的更新后的 `tcr_pmhc` 分类器类（d_model=1280，新的 MLP 维度）
- 用 ESM2 替换 embedding 提取
- 移除模型加载时的 DataParallel 包装
- 使用 `utils.py` 中的 `extract_esm2_embeddings()`
- 排名/评分逻辑保持不变

## 第 5 步：验证

1. **快速冒烟测试**：运行训练 `--max_epoch 3`，使用少量数据子集，验证：
   - ESM2 正确加载并提取 embedding
   - 分类器前向/反向传播正常
   - SwanLab 日志正常记录
   - CUDA 设备 7 上无错误

2. **完整训练运行**：运行完整训练并监控：
   - 训练损失下降
   - 验证损失下降
   - SwanLab 面板显示清晰的学习曲线

3. **预测测试**：在测试集上运行预测脚本，验证输出 CSV 正确生成

## 关键文件

- `train_tcr_pmhc_prediction_model.py` — 主要重写
- `predict_tcr_pmhc_binding.py` — 镜像更新以保持推理一致
- `utils.py` — 添加共享的 `extract_esm2_embeddings()` 函数
- `pytorchtools.py` — 保持不变（EarlyStopping）
- ESM2 模型配置: `/disk1/zhuhb/project/2026_tcr_llm/pretrain_model/esm2_t33_650M_UR50D/config.json`
