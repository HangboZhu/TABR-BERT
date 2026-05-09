import argparse

import pandas as pd

from .config import PredictorConfig
from .predictor import TcrPmhcPredictor


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict TCR-pMHC binding"
    )
    parser.add_argument(
        "--method", default="esm2", choices=["esm2", "tcr_bert"],
        help="Embedding method: 'esm2' or 'tcr_bert' (default: esm2)",
    )
    parser.add_argument(
        "--input", required=True, help="Input CSV with peptide, allele, cdr3 columns"
    )
    parser.add_argument(
        "--output", default="./output/output.csv", help="Output CSV path"
    )
    parser.add_argument("--healthy_tcr", default="")
    parser.add_argument("--allele_pseudo_seq", default="")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to tcr_pmhc_model.pt"
    )

    # ESM2 args
    parser.add_argument("--esm2_model_dir", default="", help="ESM2 model directory")
    parser.add_argument("--d_model", type=int, default=1280, help="ESM2 hidden size")

    # TCR-BERT args
    parser.add_argument("--tcr_model", default="", help="TCR-BERT TCR encoder weights")
    parser.add_argument("--pmhc_model", default="", help="TCR-BERT pMHC encoder weights")
    parser.add_argument("--bert_d_model", type=int, default=256, help="BERT hidden size")

    # Common args
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--predict_batch_size", type=int, default=1)
    parser.add_argument("--num_healthy_tcrs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    config = PredictorConfig(
        method=args.method,
        checkpoint_path=args.checkpoint,
        healthy_tcr_path=args.healthy_tcr,
        allele_pseudo_seq_path=args.allele_pseudo_seq,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        predict_batch_size=args.predict_batch_size,
        num_healthy_tcrs=args.num_healthy_tcrs,
        seed=args.seed,
        # ESM2
        esm2_model_dir=args.esm2_model_dir,
        d_model=args.d_model,
        # TCR-BERT
        tcr_model_path=args.tcr_model,
        pmhc_model_path=args.pmhc_model,
        bert_d_model=args.bert_d_model,
    )

    predictor = TcrPmhcPredictor(config)
    input_df = pd.read_csv(args.input)
    result_df = predictor.predict(input_df)
    result_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
