import mhcnames
import numpy as np
import pandas as pd
import torch

from .config import PredictorConfig
from .esm2 import ESM2Encoder
from .model import TcrPmhcClassifier


def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class TcrPmhcPredictor:
    def __init__(self, config: PredictorConfig):
        self.config = config
        self._device = torch.device(config.device)
        self._esm2_encoder: ESM2Encoder | None = None
        self._classifier: TcrPmhcClassifier | None = None
        self._allele_dict: pd.DataFrame | None = None

    def _ensure_esm2(self) -> ESM2Encoder:
        if self._esm2_encoder is None:
            self._esm2_encoder = ESM2Encoder(
                model_dir=self.config.esm2_model_dir,
                device=self._device,
                use_fp16=True,
            )
        return self._esm2_encoder

    def _ensure_classifier(self) -> TcrPmhcClassifier:
        if self._classifier is None:
            cfg = self.config
            model = TcrPmhcClassifier(
                d_model=cfg.d_model,
                tcr_maxlen=cfg.tcr_maxlen,
                pmhc_maxlen=cfg.pmhc_maxlen,
            )
            state_dict = torch.load(cfg.checkpoint_path, map_location=self._device)
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()
            self._classifier = model
        return self._classifier

    def _ensure_allele_dict(self) -> pd.DataFrame:
        if self._allele_dict is None:
            df = pd.read_csv(self.config.allele_pseudo_seq_path)
            self._allele_dict = df.set_index("allele")
        return self._allele_dict

    def _extract_healthy_tcrs(self) -> torch.Tensor:
        cfg = self.config
        healthy_df = pd.read_csv(cfg.healthy_tcr_path, nrows=cfg.num_healthy_tcrs)
        healthy_seqs = healthy_df["cdr3"].tolist()
        esm2 = self._ensure_esm2()
        return esm2.extract_embeddings(
            sequences=healthy_seqs,
            max_length=cfg.tcr_maxlen,
            d_model=cfg.d_model,
            batch_size=cfg.embedding_batch_size,
        )

    def _resolve_pmhc_sequences(
        self, peptides: list[str], alleles: list[str]
    ) -> list[str]:
        allele_dict = self._ensure_allele_dict()
        normalized = [mhcnames.normalize_allele_name(a) for a in alleles]
        try:
            pseudo_seqs = [allele_dict.at[a, "sequence"] for a in normalized]
        except KeyError as e:
            raise ValueError(
                f"Allele '{e.args[0]}' not found in pseudo-sequence file "
                f"({self.config.allele_pseudo_seq_path})."
            ) from e
        return [ps + p for ps, p in zip(pseudo_seqs, peptides)]

    def _compute_ranks(
        self,
        tcr_emb: torch.Tensor,
        pmhc_emb: torch.Tensor,
        healthy_emb: torch.Tensor,
    ) -> list[float]:
        classifier = self._ensure_classifier()
        device = self._device
        n_samples = tcr_emb.shape[0]
        n_neg = healthy_emb.shape[0]
        pool_size = n_neg + 1
        batch_size = self.config.predict_batch_size

        ranks = []
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_tcr = tcr_emb[start:end].to(device)
                batch_pmhc = pmhc_emb[start:end].to(device)
                neg_tcrs = healthy_emb.to(device)

                n_batch = end - start
                tcr_pos = batch_tcr.unsqueeze(1)  # (B, 1, tcr_dim)
                neg_expanded = neg_tcrs.unsqueeze(0).expand(n_batch, -1, -1)  # (B, N, tcr_dim)
                tcr_pool = torch.cat([tcr_pos, neg_expanded], dim=1)  # (B, N+1, tcr_dim)

                pmhc_repeated = batch_pmhc.unsqueeze(1).expand(-1, pool_size, -1)  # (B, N+1, pmhc_dim)

                tcr_flat = tcr_pool.reshape(n_batch * pool_size, -1)
                pmhc_flat = pmhc_repeated.reshape(n_batch * pool_size, -1)

                prediction = classifier(torch.cat([tcr_flat, pmhc_flat], dim=1))
                prediction = prediction.reshape(n_batch, pool_size)  # (B, N+1)

                for j in range(n_batch):
                    pred_list = prediction[j].tolist()
                    rank = 1 - (sorted(pred_list).index(pred_list[0]) + 1) / pool_size
                    ranks.append(1 - rank)

        return ranks

    def predict(self, data: pd.DataFrame | list[dict]) -> pd.DataFrame:
        _set_seed(self.config.seed)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        required = {"peptide", "allele", "cdr3"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input data missing required columns: {missing}")

        cfg = self.config

        peptides = df["peptide"].tolist()
        alleles = df["allele"].tolist()
        cdr3s = df["cdr3"].tolist()
        pmhc_sequences = self._resolve_pmhc_sequences(peptides, alleles)

        esm2 = self._ensure_esm2()
        tcr_emb = esm2.extract_embeddings(
            sequences=cdr3s,
            max_length=cfg.tcr_maxlen,
            d_model=cfg.d_model,
            batch_size=cfg.embedding_batch_size,
        )
        pmhc_emb = esm2.extract_embeddings(
            sequences=pmhc_sequences,
            max_length=cfg.pmhc_maxlen,
            d_model=cfg.d_model,
            batch_size=cfg.embedding_batch_size,
        )

        healthy_emb = self._extract_healthy_tcrs()

        esm2.unload()

        ranks = self._compute_ranks(tcr_emb, pmhc_emb, healthy_emb)
        df["rank"] = ranks

        return df
