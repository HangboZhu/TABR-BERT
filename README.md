
# TABR-BERT

## Introduction
TABR-BERT: an Accurate and Robust BERT-based Transfer Learning Model for TCR-pMHC Interaction Prediction
Publication: https://doi.org/10.1093/bib/bbad436
Contract: hui.yao@freshwindbiotech.com

## Installation

### 1. pip 安装（推荐，ESM2 版本）

使用 ESM2 替代原始 TCR-BERT/pMHC-BERT 作为特征提取器，性能更优，使用更简便。

```bash
pip install -e .
```

依赖：torch, transformers, pandas, mhcnames, numpy, scikit-learn

### 2. Docker（原始 BERT 版本）

The Installation of Docker can be seen in https://docs.docker.com/

>docker pull freshwindbioinformatics/tabr-bert:v1

>docker run -it --gpus all freshwindbioinformatics/tabr-bert:v1 bash

#### * Note : The parameter "--gpus" requires docker version higher than 19.03.

### 3. Conda and pip（原始 BERT 版本）

#### Dependencies

-   python == 3.9.12
-   mhcnames == 0.4.8
-   numpy == 1.21.5
-   pandas == 1.2.0
-   scikit_learn == 1.1.3
-   scipy == 1.8.0
-   torch == 1.11.0

Command:

> conda create -n tabr_bert python==3.9.12
> conda activate tabr_bert
> pip install -r requirements.txt

<br/>

## Data

You can find the data used to train TCR-BERT, pMHC-BERT and healthy TCR dataset at https://zenodo.org/record/8215354

## Usage (ESM2 版本)

安装后可通过 Python API 或命令行使用，详见 [tcr_pmhc_predictor/README.md](tcr_pmhc_predictor/README.md)。

### Predict（Python API）

```python
from tcr_pmhc_predictor import predict

result = predict(
    data=[
        {"peptide": "AAGIGILTV", "allele": "HLA-A*02:01", "cdr3": "CASSLSFGTEAFF"},
    ],
    esm2_model_dir="/path/to/esm2_t33_650M_UR50D/",
    checkpoint_path="./output/tcr_pmhc_model.pt",
    device="cuda:0",
)
print(result["rank"])
```

多次预测可使用 `TcrPmhcPredictor` 类复用模型：

```python
from tcr_pmhc_predictor import PredictorConfig, TcrPmhcPredictor

config = PredictorConfig(
    esm2_model_dir="/path/to/esm2_t33_650M_UR50D/",
    checkpoint_path="./output/tcr_pmhc_model.pt",
)
predictor = TcrPmhcPredictor(config)
result = predictor.predict(df)
```

### Predict（CLI）

```bash
tcr-pmhc-predict \
  --input ./data/test_S1_tcr_pmhc.csv \
  --checkpoint ./output/tcr_pmhc_model.pt \
  --esm2_model_dir /path/to/esm2_t33_650M_UR50D/ \
  --device cuda:0
```

### Rank 打分机制

预测输出不是模型的原始分数，而是一个基于排名的校准分数（rank），计算过程如下：

```
对于每条输入样本（TCR + pMHC）:

1. 将查询 TCR 与 1000 条来自健康人的背景 TCR 分别和同一个 pMHC 配对
   → 共 1001 个 TCR-pMHC 对

2. 分类模型对 1001 个对分别打分（Tanh 输出，范围 -1 ~ 1）

3. 将 1001 个分数从低到高排序，找到查询 TCR 所在的位置（position）

4. rank = 1 - (position / 1001)
```

- **rank 接近 1**：查询 TCR 的结合能力远高于随机背景 TCR，预测为强结合
- **rank 接近 0**：查询 TCR 与随机背景 TCR 无明显差异，预测为不结合

使用排名而非原始分数的原因：原始分数的绝对值缺乏直观含义，而排名表示"该 TCR 的结合能力超过了 xx% 的随机 TCR"，具有跨样本可比性。

### Train（ESM2 版本）

```bash
python train_tcr_pmhc_prediction_model_new.py \
  --input ./data/all_tcr_pmhc.csv \
  --healthy_tcr ./data/small_healthy_tcr.csv \
  --pseudo_sequence_dict ./data/mhcflurry.allele_sequences_homo.csv \
  --esm2_model_dir /path/to/esm2_t33_650M_UR50D/ \
  --gpu 0
```

## Usage (原始 BERT 版本)

### Train
#### *Note : If you don't have a GPU, then you can only run the predict file.

#### 1. pretrain TCR embedding model (TCR-BERT)

```
Usage: pre_train_tcr_embedding_model.py [options]
Required:
      --input STRING: The input data to train the TCR embedding model (*.csv)
                      Required columns: "cdr3"
      --model_dir STRING: where to save the model (*.pt)

Optional:
      --n_layers INT: number of transformer encoder layers (default: 4)
      --d_model INT: number of embedding dimention (default: 256)
      --batchsize INT: mini batchsize (default: 1024)
      --lr Float: learning rate (default: 5e-5)
      --max_epoch INT: Maximum number of train epoch (default: 100)
      --GPUs INT: num of GPUs used in this task(default: 2)
```

#### *Note : If you use docker, then you can train the TCR embedding model directly with the following command:

>python pre_train_tcr_embedding_model.py

This requires two GPUs with more than 8G of memory.

#### 2. pretrain pMHC embedding model (pMHC-BERT)

```
Usage: pre_train_pmhc_embedding_model.py [options]
Required:
      --input STRING: The input data to train the pMHC embedding model (*.csv)
                      Required columns: ["allele", "peptide", "label"]
      --random_peptide STRING: natural peptides for generating negative cases (*.csv)
                               Required columns: "peptide"
      --model_dir STRING: where to save the model (*.pt)

Optional:
      --n_layers INT: number of transformer encoder layers (default: 4)
      --d_model INT: number of embedding dimention (default: 256)
      --neg_X INT: negative case multiple (default: 2)
      --batchsize INT: mini batchsize (default: 1024)
      --lr Float: learning rate (default: 5e-5)
      --max_epoch INT: Maximum number of train epoch (default: 100)
      --GPUs INT: num of GPUs used in this task(default: 2)
```

#### *Note : If you use docker, then you can train the pMHC embedding model directly with the following command:

>python pre_train_pmhc_embedding_model.py

This requires two GPUs with more than 14G of memory.

#### 3. TCR-pMHC prediction model

```
Usage: train_tcr_pmhc_prediction_model.py [options]
Required:
      --input STRING: The input data to train the TCR-pMHC prediction model (*.csv)
                      Required columns: ["allele", "peptide", "cdr3"]
      --healthy_tcr STRING: TCRs from healthy people for generating negative cases (*.csv)
                            Required columns: "cdr3"
      --pseudo_sequence_dict STRING: allele name to pseudo sequence (*.csv)
                                     Required columns: ["allele" "sequence"]
      --tcr_model STRING: TCR embedding model dir (*.pt)
      --pmhc_model STRING: pMHC embedding model dir (*.pt)
      --model_dir STRING: where to save the model (*.pt)

Optional:
      --batchsize INT: mini batchsize (default: 256)
      --embedding_batchsize INT: mini batchsize of generation embedding (default: 256)
      --pmhc_d_model INT: dimention of pmhc embedding (default: 256)
      --tcr_d_model INT: dimention of pmhc embedding (default: 256)
      --lr Float: learning rate (default: 5e-4)
      --max_epoch INT: Maximum number of train epoch (default: 500)
      --GPUs INT: num of GPUs used in this task(default: 2)
```

#### *Note : If you use docker, then you can train the TCR-pMHC prediction model directly with the following command:

>python train_tcr_pmhc_prediction_model.py

This requires two GPUs with more than 5G of memory.

### Predict (原始版本)
```
Usage: predict_tcr_pmhc_binding.py [options]
Required:
      --input STRING: The data to be predicted (*.csv)
                      Required columns: ["allele", "peptide", "cdr3"]
      --healthy_tcr STRING: TCRs from healthy people for generating negative cases (*.csv)
                            Required columns: "cdr3"
      --pseudo_sequence_dict STRING: allele name to pseudo sequence (*.csv)
                                     Required columns: ["allele" "sequence"]
      --tcr_pmhc_model STRING: TCR-pMHC prediction model dir (*.pt)
      --tcr_model STRING: TCR embedding model dir (*.pt)
      --pmhc_model STRING: pMHC embedding model dir (*.pt)
      --output STRING: output file dir (*.csv)

Optional:
      --batchsize INT: mini batchsize (default: 256)
      --embedding_batchsize INT: mini batchsize of generation embedding (default: 256)
      --pmhc_d_model INT: dimention of pmhc embedding (default: 256)
      --tcr_d_model INT: dimention of pmhc embedding (default: 256)
      --GPUs INT: num of GPUs used in this task [if you have GPU recommend 1, if not, recommend 0] (default: 0)
```

#### *Note : If you use docker, then you can predict directly with the following command:

>python predict_tcr_pmhc_binding.py --input **input_data.csv**

## Citation

Jiawei Zhang, Wang Ma, Hui Yao, "Accurate TCR-pMHC interaction prediction using a BERT-based transfer learning method", Briefings in Bioinformatics, Volume 25, Issue 1, January 2024, bbad436, https://doi.org/10.1093/bib/bbad436
