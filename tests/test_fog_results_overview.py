from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fog_results_overview import update_overview


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_update_overview_enriches_per_class_and_confusion_matrix(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "experiment"
    fold1 = output_dir / "loso_subject_01"
    fold2 = output_dir / "loso_subject_02"

    per_class_header = "class,precision,recall_sensitivity,specificity,f1,support,tp,fp,tn,fn\n"
    write_text(
        fold1 / "per_class_metrics_test.csv",
        per_class_header
        + "0,0.8,0.9,0.7,0.85,10,9,2,20,1\n"
        + "1,0.5,0.25,0.9,0.3333333333,4,1,1,25,3\n"
        + "2,0.75,0.8,0.8,0.7741935484,10,8,2,20,2\n",
    )
    write_text(
        fold2 / "per_class_metrics_test.csv",
        per_class_header
        + "0,0.7,0.8,0.7,0.7466666667,10,8,3,19,2\n"
        + "1,0.6,0.75,0.9,0.6666666667,2,1,1,27,1\n"
        + "2,0.8,0.6,0.8,0.6857142857,10,6,1,23,4\n",
    )
    write_text(fold1 / "confusion_matrix_test.csv", ",0,1,2\n0,9,1,0\n1,1,1,2\n2,0,2,8\n")
    write_text(fold2 / "confusion_matrix_test.csv", ",0,1,2\n0,8,2,0\n1,0,1,1\n2,0,4,6\n")
    write_text(
        output_dir / "loso_summary.json",
        json.dumps(
            {
                "num_folds": 2,
                "aggregate": {
                    "test_f1_macro": {"mean": 0.4, "std": 0.1},
                    "test_recall_macro": {"mean": 0.5, "std": 0.1},
                    "test_pr_auc_macro": {"mean": 0.6, "std": 0.1},
                },
            }
        ),
    )

    overview_csv = tmp_path / "fog_results_overview.csv"
    update_overview(
        overview_csv,
        {
            "experiment": "experiment",
            "status": "ok",
            "output_dir": str(output_dir),
            "summary_path": str(output_dir / "loso_summary.json"),
        },
        sweep="unit",
    )

    with overview_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert float(row["test_f1_macro_mean"]) == 0.4
    assert float(row["test_recall_macro_mean"]) == 0.5
    assert float(row["test_pr_auc_macro_mean"]) == 0.6
    assert float(row["pre_fog_recall_mean"]) == 0.5
    assert round(float(row["pre_fog_f1_mean"]), 4) == 0.5
    assert int(row["pre_fog_support_sum"]) == 6
    assert json.loads(row["confusion_matrix_test_sum"]) == [[17, 3, 0], [1, 2, 3], [0, 6, 14]]
    assert int(row["cm_true_pre_fog_pred_fog"]) == 3
