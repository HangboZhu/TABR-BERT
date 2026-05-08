import argparse
import glob

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from tcr_pmhc_predictor import PredictorConfig, TcrPmhcPredictor


def evaluate(predictor, test_files):
    results = []
    for f in sorted(test_files):
        name = f.split("/")[-1].replace("_tcr_pmhc.csv", "")
        print(f"Evaluating {name} ...")

        df = pd.read_csv(f)
        result = predictor.predict(df)

        y_true = df["label"].values
        y_score = result["rank"].values
        y_pred = (y_score > 0.5).astype(int)

        metrics = {
            "Dataset": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_true, y_score),
        }
        results.append(metrics)

    summary = pd.DataFrame(results).set_index("Dataset")
    print("\n" + summary.to_string(float_format="%.4f"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TCR-pMHC predictor on test sets S1-S4")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--esm2_model_dir",
        type=str,
        default="/disk1/zhuhb/project/2026_tcr_llm/pretrain_model/esm2_t33_650M_UR50D/",
    )
    parser.add_argument("--checkpoint", type=str, default="./output/tcr_pmhc_model.pt")
    args = parser.parse_args()

    config = PredictorConfig(
        esm2_model_dir=args.esm2_model_dir,
        checkpoint_path=args.checkpoint,
        device=f"cuda:{args.gpu}",
    )
    predictor = TcrPmhcPredictor(config)

    test_files = sorted(glob.glob("./data/test_S*_tcr_pmhc.csv"))
    if not test_files:
        raise FileNotFoundError("No test files found matching ./data/test_S*_tcr_pmhc.csv")

    evaluate(predictor, test_files)
