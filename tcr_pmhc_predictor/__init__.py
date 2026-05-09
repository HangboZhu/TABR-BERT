import pandas as pd

from .config import PredictorConfig
from .predictor import TcrPmhcPredictor

__all__ = ["PredictorConfig", "TcrPmhcPredictor", "predict"]


def predict(
    data: pd.DataFrame | list[dict],
    method: str = "esm2",
    checkpoint_path: str = "",
    esm2_model_dir: str = "",
    tcr_model_path: str = "",
    pmhc_model_path: str = "",
    healthy_tcr_path: str = "",
    allele_pseudo_seq_path: str = "",
    device: str = "cuda:0",
    d_model: int = 1280,
    bert_d_model: int = 256,
    embedding_batch_size: int = 256,
    predict_batch_size: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    config = PredictorConfig(
        method=method,
        checkpoint_path=checkpoint_path,
        esm2_model_dir=esm2_model_dir,
        tcr_model_path=tcr_model_path,
        pmhc_model_path=pmhc_model_path,
        healthy_tcr_path=healthy_tcr_path,
        allele_pseudo_seq_path=allele_pseudo_seq_path,
        device=device,
        d_model=d_model,
        bert_d_model=bert_d_model,
        embedding_batch_size=embedding_batch_size,
        predict_batch_size=predict_batch_size,
        seed=seed,
    )
    predictor = TcrPmhcPredictor(config)
    return predictor.predict(data)
