# tcr_pmhc_predictor

TCR-pMHC 结合预测推理包，支持两种特征提取方法：
- **ESM2**：基于 ESM2 蛋白质语言模型（d_model=1280）
- **TCR-BERT**：基于原始 TABR-BERT 的自定义 BERT 模型（d_model=256）

通过 `method` 参数切换两种方法。

## 安装

```bash
pip install -e .
```

依赖：torch, transformers, pandas, mhcnames, numpy, scikit-learn

## 前置准备

### ESM2 方法

| 文件 | 说明 |
|------|------|
| ESM2 模型目录 | HuggingFace 格式的 `esm2_t33_650M_UR50D`，需要提供路径 |
| 训练好的分类器权重 | `tcr_pmhc_model.pt`（ESM2 版，约 6MB），需要提供路径 |

### TCR-BERT 方法

| 文件 | 说明 |
|------|------|
| TCR 编码器权重 | `tcr_model.pt`（约 76MB），需要提供路径 |
| pMHC 编码器权重 | `pmhc_model.pt`（约 76MB），需要提供路径 |
| 分类器权重 | `tcr_pmhc_model.pt`（BERT 版，约 586KB），需要提供路径 |

Allele 伪序列、Healthy TCR 数据和 BLOSUM 矩阵已内置在包中，无需额外准备。如需使用自定义数据，可通过参数覆盖。

## 快速使用

### 方式一：predict() 函数

适合单次预测，直接传入数据即可：

#### ESM2 方法

```python
from tcr_pmhc_predictor import predict

result = predict(
    data=[
        {"peptide": "AAGIGILTV", "allele": "HLA-A*02:01", "cdr3": "CASSLSFGTEAFF"},
        {"peptide": "GILGFVFTL", "allele": "HLA-A*02:01", "cdr3": "CASSLGQAYEQYF"},
    ],
    method="esm2",
    esm2_model_dir="/path/to/esm2_t33_650M_UR50D/",
    checkpoint_path="./output/tcr_pmhc_model.pt",
    device="cuda:0",
)

print(result[["cdr3", "allele", "peptide", "rank"]])
```

#### TCR-BERT 方法

```python
from tcr_pmhc_predictor import predict

result = predict(
    data=[
        {"peptide": "AAGIGILTV", "allele": "HLA-A*02:01", "cdr3": "CASSLSFGTEAFF"},
        {"peptide": "GILGFVFTL", "allele": "HLA-A*02:01", "cdr3": "CASSLGQAYEQYF"},
    ],
    method="tcr_bert",
    checkpoint_path="./model/tcr_pmhc_model.pt",
    tcr_model_path="./model/tcr_model.pt",
    pmhc_model_path="./model/pmhc_model.pt",
    device="cuda:0",
)
```

也支持传入 DataFrame：

```python
import pandas as pd
from tcr_pmhc_predictor import predict

df = pd.read_csv("test.csv")  # 需包含 peptide, allele, cdr3 列

# ESM2
result = predict(
    data=df,
    method="esm2",
    esm2_model_dir="/path/to/esm2_t33_650M_UR50D/",
    checkpoint_path="./output/tcr_pmhc_model.pt",
)

# TCR-BERT
result = predict(
    data=df,
    method="tcr_bert",
    checkpoint_path="./model/tcr_pmhc_model.pt",
    tcr_model_path="./model/tcr_model.pt",
    pmhc_model_path="./model/pmhc_model.pt",
)
```

### 方式二：TcrPmhcPredictor 类

适合多次预测，模型和 embedding 会被缓存复用：

```python
from tcr_pmhc_predictor import PredictorConfig, TcrPmhcPredictor

# ESM2
config = PredictorConfig(
    method="esm2",
    esm2_model_dir="/path/to/esm2_t33_650M_UR50D/",
    checkpoint_path="./output/tcr_pmhc_model.pt",
    device="cuda:0",
)

# TCR-BERT
config = PredictorConfig(
    method="tcr_bert",
    checkpoint_path="./model/tcr_pmhc_model.pt",
    tcr_model_path="./model/tcr_model.pt",
    pmhc_model_path="./model/pmhc_model.pt",
    device="cuda:0",
)

predictor = TcrPmhcPredictor(config)
result1 = predictor.predict(batch1_df)
result2 = predictor.predict(batch2_df)  # 复用已加载的模型
```

### 方式三：命令行

#### ESM2 方法

```bash
tcr-pmhc-predict \
  --method esm2 \
  --input ./data/test_S1_tcr_pmhc.csv \
  --checkpoint ./output/tcr_pmhc_model.pt \
  --esm2_model_dir /path/to/esm2_t33_650M_UR50D/ \
  --output ./output/result.csv \
  --device cuda:0
```

#### TCR-BERT 方法

```bash
tcr-pmhc-predict \
  --method tcr_bert \
  --input ./data/test_S1_tcr_pmhc.csv \
  --checkpoint ./model/tcr_pmhc_model.pt \
  --tcr_model ./model/tcr_model.pt \
  --pmhc_model ./model/pmhc_model.pt \
  --output ./output/result.csv \
  --device cuda:0
```

CLI 参数：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--method` | 否 | `esm2` | 特征提取方法：`esm2` 或 `tcr_bert` |
| `--input` | 是 | - | 输入 CSV，需包含 peptide, allele, cdr3 列 |
| `--checkpoint` | 是 | - | 分类器权重路径 |
| `--output` | 否 | `./output/output.csv` | 输出 CSV 路径 |
| `--device` | 否 | `cuda:0` | 计算设备 |
| **ESM2 专用** | | | |
| `--esm2_model_dir` | 是* | - | ESM2 模型目录路径 |
| `--d_model` | 否 | 1280 | ESM2 隐藏层维度 |
| **TCR-BERT 专用** | | | |
| `--tcr_model` | 是* | - | TCR 编码器权重路径 |
| `--pmhc_model` | 是* | - | pMHC 编码器权重路径 |
| `--bert_d_model` | 否 | 256 | BERT 隐藏层维度 |
| **通用参数** | | | |
| `--healthy_tcr` | 否 | 包内数据 | Healthy TCR 文件 |
| `--allele_pseudo_seq` | 否 | 包内数据 | MHC 伪序列文件 |
| `--embedding_batch_size` | 否 | 256 | Embedding 提取的批大小 |
| `--predict_batch_size` | 否 | 1 | 排名推理的批大小，越大越快，显存占用越高 |
| `--num_healthy_tcrs` | 否 | 1000 | 排名计算用的 healthy TCR 数量 |

*标注"是*"的参数在使用对应方法时必填。

## 输入格式

CSV 文件需包含以下三列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `peptide` | 肽段氨基酸序列 | `AAGIGILTV` |
| `allele` | MHC 等位基因名称 | `HLA-A*02:01` |
| `cdr3` | TCR CDR3 氨基酸序列 | `CASSLSFGTEAFF` |

## 输出

返回的 DataFrame 在原始数据基础上增加 `rank` 列（0~1 浮点数）：
- **rank 越接近 1**：预测结合能力越强
- **rank 越接近 0**：预测结合能力越弱

## 配置参数

通过 `PredictorConfig` 可自定义所有参数：

```python
from tcr_pmhc_predictor import PredictorConfig

config = PredictorConfig(
    method="esm2",             # 特征提取方法："esm2" 或 "tcr_bert"
    checkpoint_path="...",     # 分类器权重路径（必填）

    # ESM2 专用
    esm2_model_dir="...",      # ESM2 模型目录
    d_model=1280,              # ESM2 隐藏层维度

    # TCR-BERT 专用
    tcr_model_path="...",      # TCR 编码器权重路径
    pmhc_model_path="...",     # pMHC 编码器权重路径
    bert_d_model=256,          # BERT 隐藏层维度

    # 通用
    healthy_tcr_path="...",    # Healthy TCR 文件路径（默认使用包内数据）
    allele_pseudo_seq_path="...",  # MHC 伪序列文件路径（默认使用包内数据）
    tcr_maxlen=30,             # TCR 最大序列长度
    pmhc_maxlen=54,            # pMHC 最大序列长度
    embedding_batch_size=256,  # Embedding 提取的批大小
    predict_batch_size=1,      # 排名推理的批大小
    num_healthy_tcrs=1000,     # 排名用 healthy TCR 数量
    device="cuda:0",           # 计算设备
)
```

## 两种方法对比

| | ESM2 | TCR-BERT |
|---|---|---|
| 特征维度 | 1280 | 256 |
| 需要的权重文件 | esm2 模型 + 分类器 | TCR 编码器 + pMHC 编码器 + 分类器 |
| GPU 显存占用 | 较高（650M 参数模型） | 较低（轻量自定义 BERT） |
| 依赖 | HuggingFace transformers | 无额外依赖（模型内嵌） |
