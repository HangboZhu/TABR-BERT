import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Amino-acid index mapping (from BLOSUM matrix)
# ---------------------------------------------------------------------------

def _read_blosum_aa(path: str) -> tuple[list, dict[str, int]]:
    with open(path, "r") as f:
        blosums = []
        aa: dict[str, int] = {}
        index = 0
        for line in f:
            blosum_list: list[float] = []
            line = re.sub(r"\n", "", line)
            for info in re.split(r"\s+", line):
                try:
                    blosum_list.append(float(info))
                except ValueError:
                    if info not in aa and info.isalpha():
                        aa[info] = index
                        index += 1
            if len(blosum_list) > 0:
                blosums.append(blosum_list)
    return blosums, aa


_BLOSUM_PATH = str(Path(__file__).parent / "data" / "blosum.txt")
BLOSUMS, AA = _read_blosum_aa(_BLOSUM_PATH)


def _aa_to_index(sequence: str) -> list[int]:
    return [AA[aa] for aa in sequence]


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

_PAD_TOKEN = 22
_SEP_TOKEN = 23


def _get_attn_pad_mask(seq_q: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_q.data.eq(_PAD_TOKEN).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model * 2)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        return torch.matmul(attn, V)


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model * 2 * n_heads)
        self.W_K = nn.Linear(d_model, d_model * 2 * n_heads)
        self.W_V = nn.Linear(d_model, d_model * 2 * n_heads)
        self.d_model = d_model
        self.n_heads = n_heads
        self.liner = nn.Linear(n_heads * self.d_model * 2, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_model * 2).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_model * 2).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_model * 2).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context = _ScaledDotProductAttention(self.d_model)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model * 2)
        output = self.liner(context)
        return self.layernorm(output + residual)


class _PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.fc2(_gelu(self.fc1(x)))


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.enc_self_attn = _MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.pos_ffn = _PoswiseFeedForwardNet(d_model=d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        return self.pos_ffn(enc_outputs)


# ---------------------------------------------------------------------------
# TCR BERT (no segment embedding)
# ---------------------------------------------------------------------------

class _TCREmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, maxlen: int, device: torch.device):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(x)
        return self.norm(self.tok_embed(x) + self.pos_embed(pos))


class BERT_TCR(nn.Module):
    def __init__(self, n_layers: int = 4, d_model: int = 256, n_heads: int = 8,
                 vocab_size: int = 23, maxlen: int = 30, device: torch.device = None):
        super().__init__()
        self.d_model = d_model
        self.device = device or torch.device("cpu")
        self.embedding = _TCREmbedding(d_model, vocab_size, maxlen, self.device)
        self.layers = nn.ModuleList([_EncoderLayer(d_model, n_heads) for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = self.embedding.tok_embed.weight

    def forward(self, input_ids):
        output = self.embedding(input_ids)
        enc_self_attn_mask = _get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        return output


# ---------------------------------------------------------------------------
# pMHC BERT (with segment embedding)
# ---------------------------------------------------------------------------

class _PMHCEmbedding(nn.Module):
    def __init__(self, d_model: int, maxlen: int, n_segments: int, vocab_size: int, device: torch.device):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(x)
        return self.norm(self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg))


class BERT_PMHC(nn.Module):
    def __init__(self, n_layers: int = 4, d_model: int = 256, n_heads: int = 8,
                 maxlen: int = 54, n_segments: int = 2, vocab_size: int = 25,
                 device: torch.device = None):
        super().__init__()
        self.d_model = d_model
        self.device = device or torch.device("cpu")
        self.embedding = _PMHCEmbedding(d_model, maxlen, n_segments, vocab_size, self.device)
        self.layers = nn.ModuleList([_EncoderLayer(d_model, n_heads) for _ in range(n_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Dropout(p=0.2),
            nn.SELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.linear = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = self.embedding.tok_embed.weight

    def forward(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = _get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        return output


# ---------------------------------------------------------------------------
# Encoder wrapper (same interface style as ESM2Encoder)
# ---------------------------------------------------------------------------

class TCRBERTEncoder:
    def __init__(self, tcr_model_path: str, pmhc_model_path: str, device: torch.device):
        self._tcr_model_path = tcr_model_path
        self._pmhc_model_path = pmhc_model_path
        self._device = device
        self._tcr_model: BERT_TCR | None = None
        self._pmhc_model: BERT_PMHC | None = None

    def _load(self):
        if self._tcr_model is not None:
            return

        tcr = BERT_TCR(device=self._device)
        state_dict = torch.load(self._tcr_model_path, map_location=self._device)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        tcr.load_state_dict(state_dict)
        tcr.to(self._device)
        tcr.eval()
        self._tcr_model = tcr

        pmhc = BERT_PMHC(device=self._device)
        state_dict = torch.load(self._pmhc_model_path, map_location=self._device)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        pmhc.load_state_dict(state_dict)
        pmhc.to(self._device)
        pmhc.eval()
        self._pmhc_model = pmhc

    @property
    def tcr_model(self) -> BERT_TCR:
        self._load()
        return self._tcr_model

    @property
    def pmhc_model(self) -> BERT_PMHC:
        self._load()
        return self._pmhc_model

    def extract_tcr_embeddings(
        self,
        sequences: list[str],
        max_length: int,
        d_model: int = 256,
        batch_size: int = 256,
    ) -> torch.Tensor:
        self._load()
        all_inputs = []
        for seq in sequences:
            ids = _aa_to_index(seq)
            ids.extend([_PAD_TOKEN] * (max_length - len(ids)))
            all_inputs.append(ids)

        loader = DataLoader(
            TensorDataset(torch.LongTensor(all_inputs)),
            batch_size=batch_size, shuffle=False,
        )

        all_emb: list[torch.Tensor] = []
        with torch.no_grad():
            for (input_ids,) in loader:
                output = self._tcr_model(input_ids.to(self._device))
                all_emb.append(output.reshape(-1, max_length * d_model).cpu())

        return torch.cat(all_emb, dim=0)

    def extract_pmhc_embeddings(
        self,
        allele_sequences: list[str],
        peptides: list[str],
        max_length: int,
        d_model: int = 256,
        batch_size: int = 256,
    ) -> torch.Tensor:
        self._load()
        all_input_ids = []
        all_segment_ids = []
        for allele, peptide in zip(allele_sequences, peptides):
            input_ids = _aa_to_index(allele) + [_SEP_TOKEN] + _aa_to_index(peptide) + [_SEP_TOKEN]
            segment_ids = [0] * (len(allele) + 1) + [1] * (len(peptide) + 1)
            n_pad = max_length - len(input_ids)
            input_ids.extend([_PAD_TOKEN] * n_pad)
            segment_ids.extend([1] * n_pad)
            all_input_ids.append(input_ids)
            all_segment_ids.append(segment_ids)

        loader = DataLoader(
            TensorDataset(torch.LongTensor(all_input_ids), torch.LongTensor(all_segment_ids)),
            batch_size=batch_size, shuffle=False,
        )

        all_emb: list[torch.Tensor] = []
        with torch.no_grad():
            for input_ids, segment_ids in loader:
                output = self._pmhc_model(input_ids.to(self._device), segment_ids.to(self._device))
                all_emb.append(output.reshape(-1, max_length * d_model).cpu())

        return torch.cat(all_emb, dim=0)

    def unload(self):
        if self._tcr_model is not None:
            del self._tcr_model
            self._tcr_model = None
        if self._pmhc_model is not None:
            del self._pmhc_model
            self._pmhc_model = None
        torch.cuda.empty_cache()
