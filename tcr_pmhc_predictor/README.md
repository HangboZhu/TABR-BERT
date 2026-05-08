# tcr_pmhc_predictor

TCR-pMHC 结合预测推理包，基于 ESM2 蛋白质语言模型提取特征，通过训练好的分类器计算 TCR 与 pMHC 的结合排名分数。

## 安装

```bash
pip install -e .
```

依赖：torch, transformers, pandas, mhcnames, numpy, scikit-learn

## 前置准备

使用前需准备好以下文件：

| 文件 | 说明 |
|------|------|
| ESM2 模型目录 | HuggingFace 格式的 `esm2_t33_650M_UR50D`，需要提供路径 |
| 训练好的分类器权重 | `tcr_pmhc_model.pt` 文件（约 6MB），需要提供路径 |

Allele 伪序列和 Healthy TCR 数据已内置在包中，无需额外准备。如需使用自定义数据，可通过参数覆盖。

## 快速使用

### 方式一：predict() 函数

适合单次预测，直接传入数据即可：

```python
from tcr_pmhc_predictor import predict

result = predict(
    data=[
        {"peptide": "AAGIGILTV", "allele": "HLA-A*02:01", "cdr3": "CASSLSFGTEAFF"},
        {"peptide": "GILGFVFTL", "allele": "HLA-A*02:01", "cdr3": "CASSLGQAYEQYF"},
    ],
    esm2_model_dir="/path/to/esm2_t33_650M_UR50D/",
    checkpoint_path="./output/tcr_pmhc_model.pt",
    device="cuda:0",
)

print(result[["cdr3", "allele", "peptide", "rank"]])
```

也支持传入 DataFrame：

```python
import pandas as pd
from tcr_pmhc_predictor import predict

df = pd.read_csv("test.csv")  # 需包含 peptide, allele, cdr3 列
result = predict(
    data=df,
    esm2_model_dir="/path/to/esm2_t33_650M_UR50D/",
    checkpoint_path="./output/tcr_pmhc_model.pt",
)
```

### 方式二：TcrPmhcPredictor 类

适合多次预测，模型和 embedding 会被缓存复用：

```python
from tcr_pmhc_predictor import PredictorConfig, TcrPmhcPredictor

config = PredictorConfig(
    esm2_model_dir="/path/to/esm2_t33_650M_UR50D/",
    checkpoint_path="./output/tcr_pmhc_model.pt",
    device="cuda:0",
)
predictor = TcrPmhcPredictor(config)

result1 = predictor.predict(batch1_df)
result2 = predictor.predict(batch2_df)  # 复用已加载的模型
```

### 方式三：命令行

```bash
tcr-pmhc-predict \
  --input ./data/test_S1_tcr_pmhc.csv \
  --checkpoint ./output/tcr_pmhc_model.pt \
  --esm2_model_dir /path/to/esm2_t33_650M_UR50D/ \
  --output ./output/result.csv \
  --device cuda:0
```

CLI 参数：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | 是 | - | 输入 CSV，需包含 peptide, allele, cdr3 列 |
| `--checkpoint` | 是 | - | 训练好的模型权重路径 |
| `--esm2_model_dir` | 是 | - | ESM2 模型目录路径 |
| `--output` | 否 | `./output/output.csv` | 输出 CSV 路径 |
| `--healthy_tcr` | 否 | 包内数据 | Healthy TCR 文件 |
| `--allele_pseudo_seq` | 否 | 包内数据 | MHC 伪序列文件 |
| `--device` | 否 | `cuda:0` | 计算设备 |
| `--embedding_batch_size` | 否 | 256 | ESM2 embedding 提取的批大小 |
| `--predict_batch_size` | 否 | 1 | 排名推理的批大小，越大越快，显存占用越高 |
| `--num_healthy_tcrs` | 否 | 1000 | 排名计算用的 healthy TCR 数量 |

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
    esm2_model_dir="...",       # ESM2 模型目录（必填）
    checkpoint_path="...",      # 分类器权重路径（必填）
    healthy_tcr_path="...",     # Healthy TCR 文件路径（默认使用包内数据）
    allele_pseudo_seq_path="...",  # MHC 伪序列文件路径（默认使用包内数据）
    d_model=1280,               # ESM2 隐藏层维度
    tcr_maxlen=30,              # TCR 最大序列长度
    pmhc_maxlen=54,             # pMHC 最大序列长度
    embedding_batch_size=256,   # ESM2 embedding 提取的批大小
    predict_batch_size=1,       # 排名推理的批大小，越大越快，显存占用越高
    num_healthy_tcrs=1000,      # 排名用 healthy TCR 数量
    device="cuda:0",            # 计算设备
)
```
