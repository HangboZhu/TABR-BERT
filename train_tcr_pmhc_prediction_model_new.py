import torch
from torch import nn
import pandas as pd
import argparse
from utils import weight_init, accuracy_func, extract_esm2_embeddings
import random
import mhcnames
import swanlab
from transformers import EsmTokenizer, EsmModel
from pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

parser = argparse.ArgumentParser(description='train the tcr-pmhc prediction model')

# File dir
parser.add_argument('--input', type=str, default="./data/all_tcr_pmhc.csv", help="input data ,includes \
                    the following three columns:'peptide','allele','cdr3'")
parser.add_argument('--healthy_tcr', type=str, default="./data/small_healthy_tcr.csv", \
                    help='TCR for healthy people csv')
parser.add_argument('--pseudo_sequence_dict', default="./data/mhcflurry.allele_sequences_homo.csv", \
                    type=str, help='allele name to pseudo sequence csv file dir')
parser.add_argument('--model_dir', type=str, default="./output/tcr_pmhc_model.pt", help='where to save the model')

# ESM2 model
parser.add_argument('--esm2_model_dir', type=str,
                    default="/disk1/zhuhb/project/2026_tcr_llm/pretrain_model/esm2_t33_650M_UR50D/",
                    help='ESM2 model directory')

# Hyperparameters
parser.add_argument('--batchsize', type=int, default=256, help='mini batchsize')
parser.add_argument('--embedding_batchsize', type=int, default=256, \
                    help='mini batchsize of generation embedding')
parser.add_argument('--d_model', type=int, default=1280, help='ESM2 hidden size')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=500)

# GPU
parser.add_argument('--gpu', type=int, default=7, help='CUDA device index')

# SwanLab
parser.add_argument('--swanlab_project', type=str, default="TABR-BERT-ESM2")
parser.add_argument('--swanlab_experiment', type=str, default="tcr_pmhc_training")

args = parser.parse_args()

input_data_file = args.input
healthy_tcr_file = args.healthy_tcr
d_model = args.d_model
pseudo_sequence_file = args.pseudo_sequence_dict
esm2_model_dir = args.esm2_model_dir
embedding_BATCH_SIZE = args.embedding_batchsize
model_dir = args.model_dir
pmhc_maxlen = 54
tcr_maxlen = 30
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)

# Loading and processing data
df = pd.read_csv(input_data_file)
df = df[df["train_test"] == "train"]
cdr3s = df["cdr3"].tolist()
alleles = df["allele"].tolist()
alleles = [mhcnames.normalize_allele_name(allele) for allele in alleles]
peptides = df["peptide"].tolist()

healthy_tcrs = pd.read_csv(healthy_tcr_file, nrows=10000)["cdr3"].tolist()

# Map allele names to pseudo-sequences
allele_dict = pd.read_csv(pseudo_sequence_file)
allele_dict = allele_dict.set_index("allele")
allele_sequences = [allele_dict.at[allele, "sequence"] for allele in alleles]

# Build pMHC concatenated sequences
pmhc_sequences = [a + p for a, p in zip(allele_sequences, peptides)]

# Load shared ESM2 model (frozen)
esm2_tokenizer = EsmTokenizer.from_pretrained(esm2_model_dir)
esm2_model = EsmModel.from_pretrained(esm2_model_dir)
esm2_model = esm2_model.half()
esm2_model.to(device)
esm2_model.eval()
for param in esm2_model.parameters():
    param.requires_grad = False

# Extract TCR embeddings
tcr_output = extract_esm2_embeddings(
    cdr3s, esm2_tokenizer, esm2_model, device, tcr_maxlen, embedding_BATCH_SIZE, d_model
)

# Extract pMHC embeddings
pmhc_output = extract_esm2_embeddings(
    pmhc_sequences, esm2_tokenizer, esm2_model, device, pmhc_maxlen, embedding_BATCH_SIZE, d_model
)

# Extract healthy TCR embeddings
healthy_output = extract_esm2_embeddings(
    healthy_tcrs, esm2_tokenizer, esm2_model, device, tcr_maxlen, embedding_BATCH_SIZE, d_model
)

# Split background TCR and negative sampling pool
TCR_neg_df_1k = healthy_output[:1000, :]
healthy_tcrs_matrix = healthy_output[1000:, :]

# Free ESM2 from GPU memory
del esm2_model
torch.cuda.empty_cache()

print("embedding generation is complete")


# TCR-pMHC prediction model
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


# Loss function
class TCR_PMHC_loss(nn.Module):
    def __init__(self):
        super(TCR_PMHC_loss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, pos_pre, pos_label, neg_pre, neg_label):
        loss = torch.mean(self.relu(1 + neg_pre - pos_pre)) + 0.2 * torch.mean(neg_pre ** 2 + pos_pre ** 2)
        return loss


def compute_metrics(preds, labels):
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    preds_binary = (preds_np > 0).astype(int)
    return {
        "accuracy": accuracy_func(preds, labels, 0),
        "precision": precision_score(labels_np, preds_binary, zero_division=0),
        "recall": recall_score(labels_np, preds_binary, zero_division=0),
        "f1": f1_score(labels_np, preds_binary, zero_division=0),
        "auc": roc_auc_score(labels_np, preds_np),
    }


def make_epoch_neg_indices(n, healthy_tcrs_matrix, device):
    return torch.randint(0, healthy_tcrs_matrix.shape[0], (n,), device=device)


def iter_gpu_batches(tcrs, pmhcs, healthy_tcrs_matrix, batch_size, neg_indices=None, shuffle=True):
    n = tcrs.shape[0]
    if shuffle:
        order = torch.randperm(n, device=tcrs.device)
    else:
        order = torch.arange(n, device=tcrs.device)

    for start in range(0, n, batch_size):
        indices = order[start:start + batch_size]
        batch_tcrs = tcrs[indices]
        batch_pmhcs = pmhcs[indices]
        if neg_indices is None:
            batch_neg_indices = torch.randint(0, healthy_tcrs_matrix.shape[0], (indices.shape[0],), device=tcrs.device)
        else:
            batch_neg_indices = neg_indices[indices]
        neg_tcrs = healthy_tcrs_matrix[batch_neg_indices]
        pos = torch.cat([batch_tcrs, batch_pmhcs], dim=-1)
        neg = torch.cat([neg_tcrs, batch_pmhcs], dim=-1)
        pos_label = torch.ones(indices.shape[0], device=tcrs.device)
        neg_label = torch.zeros(indices.shape[0], device=tcrs.device)
        yield pos, neg, pos_label, neg_label


# Run the main program
if __name__ == "__main__":
    # Split training and validation set
    train_tcr, val_tcr, train_pmhc, val_pmhc \
        = train_test_split(tcr_output, pmhc_output, test_size=0.1, random_state=0)

    train_tcr = train_tcr.to(device)
    train_pmhc = train_pmhc.to(device)
    val_tcr = val_tcr.to(device)
    val_pmhc = val_pmhc.to(device)
    healthy_tcrs_matrix = healthy_tcrs_matrix.to(device)
    del tcr_output, pmhc_output, healthy_output, TCR_neg_df_1k

    # Initialize model
    model = tcr_pmhc()
    weight_init(model)
    model.to(device)

    # Loss function
    LOSS = TCR_PMHC_loss()

    # Hyperparameters
    BATCH_SIZE = args.batchsize
    EPOCH = args.max_epoch
    lr = args.lr

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # LR scheduler and early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3,
                                                           patience=2)
    early_stopping = EarlyStopping(patience=6, verbose=True)

    # Init SwanLab
    swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment,
        config={
            "learning_rate": lr,
            "batch_size": BATCH_SIZE,
            "max_epoch": EPOCH,
            "esm2_model": "esm2_t33_650M_UR50D",
            "embedding_dim": d_model,
            "tcr_maxlen": tcr_maxlen,
            "pmhc_maxlen": pmhc_maxlen,
        }
    )

    # Create fixed validation negative samples
    val_neg_indices = make_epoch_neg_indices(val_tcr.shape[0], healthy_tcrs_matrix, device)

    for epoch in range(EPOCH):
        tra_loss = torch.zeros((), device=device)

        # Regenerate negative samples every 5 epochs
        if epoch % 5 == 0:
            train_neg_indices = make_epoch_neg_indices(train_tcr.shape[0], healthy_tcrs_matrix, device)

        # Training
        model.train()

        for tra_step, (pos, neg, pos_label, neg_label) in enumerate(
            iter_gpu_batches(train_tcr, train_pmhc, healthy_tcrs_matrix, BATCH_SIZE, train_neg_indices)
        ):

            pos_pred = model(pos)
            neg_pred = model(neg)
            loss = LOSS(pos_pred, pos_label, neg_pred, neg_label)

            tra_loss += loss.detach()

            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            if tra_step % 200 == 0:
                batch_preds = torch.cat([pos_pred, neg_pred])
                batch_labels = torch.cat([pos_label, neg_label])
                metrics = compute_metrics(batch_preds, batch_labels)
                swanlab.log({
                    "train/loss": loss.item(),
                    "train/accuracy": metrics["accuracy"],
                    "train/precision": metrics["precision"],
                    "train/recall": metrics["recall"],
                    "train/f1": metrics["f1"],
                    "train/auc": metrics["auc"],
                    "train/epoch": epoch + 1,
                    "train/step": tra_step + 1,
                })
                print(" epoch: {} step: {}  loss: {:.3f} acc: {:.3f} precision: {:.3f} recall: {:.3f} f1: {:.3f} auc: {:.3f}".format(
                    epoch + 1, tra_step + 1, loss.item(), metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["auc"]))

        # Validation
        model.eval()
        val_loss = torch.zeros((), device=device)
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for val_step, (pos, neg, pos_label, neg_label) in enumerate(
                iter_gpu_batches(val_tcr, val_pmhc, healthy_tcrs_matrix, BATCH_SIZE, val_neg_indices)
            ):

                pos_pred = model(pos)
                neg_pred = model(neg)
                loss = LOSS(pos_pred, pos_label, neg_pred, neg_label)

                batch_preds = torch.cat([pos_pred, neg_pred])
                batch_labels = torch.cat([pos_label, neg_label])

                val_loss += loss.detach()

                all_val_preds.append(batch_preds.detach())
                all_val_labels.append(batch_labels.detach())

                if val_step % 50 == 0:
                    metrics = compute_metrics(batch_preds, batch_labels)
                    swanlab.log({
                        "val/loss": loss.item(),
                        "val/accuracy": metrics["accuracy"],
                        "val/precision_step": metrics["precision"],
                        "val/recall_step": metrics["recall"],
                        "val/f1_step": metrics["f1"],
                        "val/auc_step": metrics["auc"],
                        "val/epoch": epoch + 1,
                        "val/step": val_step + 1,
                    })
                    print(" epoch: {} step: {}  loss {:.3f} acc: {:.3f} precision: {:.3f} recall: {:.3f} f1: {:.3f} auc: {:.3f}".format(
                        epoch + 1, val_step + 1, loss.item(), metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["auc"]))

        # Compute epoch-level metrics
        val_preds_all = torch.cat(all_val_preds).cpu().numpy()
        val_labels_all = torch.cat(all_val_labels).cpu().numpy()
        val_preds_binary = (val_preds_all > 0).astype(int)

        val_precision = precision_score(val_labels_all, val_preds_binary, zero_division=0)
        val_recall = recall_score(val_labels_all, val_preds_binary, zero_division=0)
        val_f1 = f1_score(val_labels_all, val_preds_binary, zero_division=0)
        val_auc = roc_auc_score(val_labels_all, val_preds_all)

        # LR scheduler
        avg_train_loss = (tra_loss / (tra_step + 1)).item()
        avg_val_loss = (val_loss / (val_step + 1)).item()
        scheduler.step(avg_val_loss)

        # Log epoch-level metrics
        swanlab.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "val/precision": val_precision,
            "val/recall": val_recall,
            "val/f1": val_f1,
            "val/auc": val_auc,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1,
        })

        print(" epoch: {}  val_loss: {:.3f} precision: {:.3f} recall: {:.3f} f1: {:.3f} auc: {:.3f}".format(
            epoch + 1, avg_val_loss, val_precision, val_recall, val_f1, val_auc))

        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping at epoch {}".format(epoch + 1))
            break

        # Save model
        torch.save(model.state_dict(), model_dir)

    swanlab.finish()
