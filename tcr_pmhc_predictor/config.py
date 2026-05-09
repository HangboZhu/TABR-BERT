import os
from dataclasses import dataclass
from pathlib import Path


def _package_data_path(filename: str) -> str:
    return str(Path(__file__).parent / "data" / filename)


@dataclass(frozen=True)
class PredictorConfig:
    # --- common ---
    method: str = "esm2"  # "esm2" or "tcr_bert"
    checkpoint_path: str = ""

    healthy_tcr_path: str = ""
    allele_pseudo_seq_path: str = ""

    tcr_maxlen: int = 30
    pmhc_maxlen: int = 54

    embedding_batch_size: int = 256
    predict_batch_size: int = 1
    num_healthy_tcrs: int = 1000
    seed: int = 0
    device: str = "cuda:0"

    # --- ESM2 specific ---
    esm2_model_dir: str = ""
    d_model: int = 1280

    # --- TCR-BERT specific ---
    tcr_model_path: str = ""
    pmhc_model_path: str = ""
    bert_d_model: int = 256

    def __post_init__(self):
        if not self.healthy_tcr_path:
            object.__setattr__(self, "healthy_tcr_path", _package_data_path("small_healthy_tcr.csv"))
        if not self.allele_pseudo_seq_path:
            object.__setattr__(self, "allele_pseudo_seq_path", _package_data_path("mhcflurry.allele_sequences_homo.csv"))
        if self.method not in ("esm2", "tcr_bert"):
            raise ValueError(f"method must be 'esm2' or 'tcr_bert', got '{self.method}'")
