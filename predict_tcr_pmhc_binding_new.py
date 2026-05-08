import torch
import torch.nn as nn
from utils import extract_esm2_embeddings
import mhcnames
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from transformers import EsmTokenizer, EsmModel

parser = argparse.ArgumentParser(description='predict whether the tcr and pmhc can bind (ESM-2 based)')

# File dir
parser.add_argument('--input', type=str, default="./data/test_S1_tcr_pmhc.csv", help="input data ,includes \
                    the following three columns:'peptide','allele','cdr3'")
parser.add_argument('--healthy_tcr', type=str, default="./data/small_healthy_tcr.csv", \
                    help='TCR for healthy people csv')
parser.add_argument('--pseudo_sequence_dict', type=str, default="./data/mhcflurry.allele_sequences_homo.csv", \
                    help='allele name to pseudo sequence csv file dir')
parser.add_argument('--output', type=str, default="./output/output.csv", \
                    help='output file')

# model weights
parser.add_argument('--tcr_pmhc_model', type=str, default="./output/tcr_pmhc_model.pt", \
                    help='TCR-pMHC prediction model dir')
parser.add_argument('--esm2_model_dir', type=str,
                    default="/disk1/zhuhb/project/2026_tcr_llm/pretrain_model/esm2_t33_650M_UR50D/",
                    help='ESM2 model directory')

# Hyperparameters
parser.add_argument('--embedding_batchsize', type=int, default=256, \
                    help='mini batchsize of generation embedding')
parser.add_argument('--d_model', type=int, default=1280, help='ESM2 hidden size')

# GPU
parser.add_argument('--gpu', type=int, default=7, help='CUDA device index')

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)

input_data_file = args.input
output_file = args.output
healthy_tcr_file = args.healthy_tcr
pseudo_sequence_file = args.pseudo_sequence_dict
tcr_pmhc_model_file = args.tcr_pmhc_model
esm2_model_dir = args.esm2_model_dir
d_model = args.d_model
embedding_BATCH_SIZE = args.embedding_batchsize
tcr_maxlen = 30
pmhc_maxlen = 54


# TCR-pMHC prediction model (same architecture as train_tcr_pmhc_prediction_model_new.py)
class tcr_pmhc(nn.Module):
    def __init__(self):
        super(tcr_pmhc, self).__init__()
        self.pmhc_linear = nn.Sequential(
            nn.Linear(d_model, 1),
        )
        self.tcr_linear = nn.Sequential(
            nn.Linear(d_model, 1),
        )

        self._pmhc_linear = nn.Sequential(
            nn.Linear(pmhc_maxlen, 1),
        )
        self._tcr_linear = nn.Sequential(
            nn.Linear(tcr_maxlen, 1),
        )

        self.dense = nn.Sequential(
            nn.Linear(pmhc_maxlen + tcr_maxlen + d_model * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape(-1, pmhc_maxlen + tcr_maxlen, d_model)
        tcr = x[:, :tcr_maxlen, :]
        pmhc = x[:, tcr_maxlen:, :]

        tcr_x = self.tcr_linear(tcr).reshape(-1, tcr_maxlen)
        pmhc_x = self.pmhc_linear(pmhc).reshape(-1, pmhc_maxlen)

        _tcr_x = self._tcr_linear(tcr.transpose(1, 2)).reshape(-1, d_model)
        _pmhc_x = self._pmhc_linear(pmhc.transpose(1, 2)).reshape(-1, d_model)

        out = self.dense(torch.cat([tcr_x, _tcr_x, pmhc_x, _pmhc_x], dim=-1))
        return out.view(-1)


# Initialize tcr-pmhc prediction model and load weights
model = tcr_pmhc()
model.load_state_dict(torch.load(tcr_pmhc_model_file, map_location=device))
model.to(device)
model.eval()

# Load ESM2 model for embedding extraction
esm2_tokenizer = EsmTokenizer.from_pretrained(esm2_model_dir)
esm2_model = EsmModel.from_pretrained(esm2_model_dir)
esm2_model = esm2_model.half()
esm2_model.to(device)
esm2_model.eval()
for param in esm2_model.parameters():
    param.requires_grad = False

# Loading and processing data
allele_dict = pd.read_csv(pseudo_sequence_file)
allele_dict = allele_dict.set_index("allele")

input_df = pd.read_csv(input_data_file)
alleles = input_df["allele"].tolist()
alleles = [mhcnames.normalize_allele_name(allele) for allele in alleles]
peptides = input_df["peptide"].tolist()
cdr3s = input_df["cdr3"].tolist()
healthy_tcrs = pd.read_csv(healthy_tcr_file, nrows=1000)["cdr3"].tolist()

allele_sequences = [allele_dict.at[allele, "sequence"] for allele in alleles]
pmhc_sequences = [a + p for a, p in zip(allele_sequences, peptides)]

# Extract embeddings using ESM2
tcr_output = extract_esm2_embeddings(
    cdr3s, esm2_tokenizer, esm2_model, device, tcr_maxlen, embedding_BATCH_SIZE, d_model
)

pmhc_output = extract_esm2_embeddings(
    pmhc_sequences, esm2_tokenizer, esm2_model, device, pmhc_maxlen, embedding_BATCH_SIZE, d_model
)

healthy_output = extract_esm2_embeddings(
    healthy_tcrs, esm2_tokenizer, esm2_model, device, tcr_maxlen, embedding_BATCH_SIZE, d_model
)

TCR_neg_df_1k = healthy_output

# Free ESM2
del esm2_model
torch.cuda.empty_cache()


def get_rank(test_tcr, test_pmhc, TCR_neg_df_1k, test_label=False):
    preds = []
    ranks = []
    for each_data_index in range(test_tcr.shape[0]):
        tcr_pos = test_tcr[each_data_index].unsqueeze(dim=0).to(device)
        pmhc = test_pmhc[each_data_index].unsqueeze(dim=0).to(device)
        neg_tcrs = TCR_neg_df_1k.to(device)

        TCR_input_df = torch.cat([tcr_pos, neg_tcrs], dim=0)
        MHC_antigen_input_df = torch.repeat_interleave(pmhc, 1001, dim=0)

        prediction = model(torch.cat([TCR_input_df, MHC_antigen_input_df], dim=1))
        preds.append(prediction.tolist()[0])
        rank = 1 - (sorted(prediction.tolist()).index(prediction.tolist()[0]) + 1) / 1001
        ranks.append(rank)
    preds = torch.tensor(preds, dtype=torch.float32)
    ranks = [1 - i for i in ranks]
    if test_label:
        precision, recall, _thresholds = precision_recall_curve(test_label, ranks)
        PR = auc(recall, precision)
        AUC = roc_auc_score(test_label, ranks)
        return ranks, PR, AUC
    else:
        return ranks


ranks = get_rank(tcr_output, pmhc_output, TCR_neg_df_1k)
input_df["rank"] = ranks
input_df.to_csv(output_file, index=False)
