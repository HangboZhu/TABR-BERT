import pandas as pd

from .config import PredictorConfig
from .predictor import TcrPmhcPredictor

__all__ = ["PredictorConfig", "TcrPmhcPredictor", "predict"]


def predict(
    data: pd.DataFrame | list[dict],
    esm2_model_dir: str,
    checkpoint_path: str,
    healthy_tcr_path: str = "",
    allele_pseudo_seq_path: str = "",
    device: str = "cuda:0",
    embedding_batch_size: int = 256,
    predict_batch_size: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    config = PredictorConfig(
        esm2_model_dir=esm2_model_dir,
        checkpoint_path=checkpoint_path,
        healthy_tcr_path=healthy_tcr_path,
        allele_pseudo_seq_path=allele_pseudo_seq_path,
        device=device,
        embedding_batch_size=embedding_batch_size,
        predict_batch_size=predict_batch_size,
        seed=seed,
    )
    predictor = TcrPmhcPredictor(config)
    return predictor.predict(data)
