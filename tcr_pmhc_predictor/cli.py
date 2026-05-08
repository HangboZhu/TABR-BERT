import argparse

import pandas as pd

from .config import PredictorConfig
from .predictor import TcrPmhcPredictor


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict TCR-pMHC binding using ESM2 embeddings"
    )
    parser.add_argument(
        "--input", required=True, help="Input CSV with peptide, allele, cdr3 columns"
    )
    parser.add_argument(
        "--output", default="./output/output.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--healthy_tcr", default=""
    )
    parser.add_argument(
        "--allele_pseudo_seq",
        default="",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to trained tcr_pmhc_model.pt"
    )
    parser.add_argument(
        "--esm2_model_dir", required=True, help="Path to ESM2 model directory"
    )
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--predict_batch_size", type=int, default=1, help="Batch size for ranking inference")
    parser.add_argument(
        "--device", default="cuda:0", help="Torch device (e.g. cuda:0, cpu)"
    )
    parser.add_argument("--num_healthy_tcrs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args(argv)

    config = PredictorConfig(
        esm2_model_dir=args.esm2_model_dir,
        checkpoint_path=args.checkpoint,
        healthy_tcr_path=args.healthy_tcr,
        allele_pseudo_seq_path=args.allele_pseudo_seq,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        predict_batch_size=args.predict_batch_size,
        num_healthy_tcrs=args.num_healthy_tcrs,
        seed=args.seed,
    )

    predictor = TcrPmhcPredictor(config)
    input_df = pd.read_csv(args.input)
    result_df = predictor.predict(input_df)
    result_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
