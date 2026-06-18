from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

from exp.exp_fog_classification import ClassificationExperiment
from exp.exp_simclr import SimCLRExperiment
from exp.exp_supcon import SupConExperiment
from exp.exp_utils import scalar_metrics, summarize, write_csv
from utils.io import load_checkpoint, save_json
from utils.distributed import barrier, is_main_process


CLASS_NAMES_BY_COUNT = {
    2: ["normal", "fog"],
    3: ["normal", "pre_fog", "fog"],
}


def _class_key(class_index: int, num_classes: int) -> str:
    names = CLASS_NAMES_BY_COUNT.get(int(num_classes), [])
    if 0 <= int(class_index) < len(names):
        return names[int(class_index)]
    return f"class_{int(class_index)}"


def _read_float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def _read_int(row: dict[str, str], *keys: str) -> int | None:
    value = _read_float(row, *keys)
    return None if value is None else int(value)


def _read_per_class_metrics(fold_output_dir: Path, num_classes: int) -> dict[str, float | int]:
    path = fold_output_dir / "per_class_metrics_test.csv"
    if not path.exists():
        return {}
    out: dict[str, float | int] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_index = int(float(row["class"]))
            prefix = f"test_{_class_key(class_index, num_classes)}"
            precision = _read_float(row, "precision")
            recall = _read_float(row, "recall_sensitivity", "recall")
            f1 = _read_float(row, "f1")
            support = _read_int(row, "support")
            if precision is not None:
                out[f"{prefix}_precision"] = precision
            if recall is not None:
                out[f"{prefix}_recall"] = recall
            if f1 is not None:
                out[f"{prefix}_f1"] = f1
            if support is not None:
                out[f"{prefix}_support"] = support
    return out


def _read_confusion_matrix(path: Path) -> list[list[int]] | None:
    if not path.exists():
        return None
    matrix: list[list[int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            values = row[1:] if len(row) > 1 else row
            matrix.append([int(float(value)) for value in values if value != ""])
    return matrix or None


def _sum_matrices(matrices: list[list[list[int]]]) -> list[list[int]] | None:
    if not matrices:
        return None
    rows = len(matrices[0])
    cols = len(matrices[0][0]) if rows else 0
    total = [[0 for _ in range(cols)] for _ in range(rows)]
    for matrix in matrices:
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            continue
        for i in range(rows):
            for j in range(cols):
                total[i][j] += int(matrix[i][j])
    return total


class LOSOExperiment:
    """Leave-one-subject-out experiment over prepared fold directories."""

    def __init__(self, cfg):
        self.cfg = cfg
        exp_cfg = cfg.get("experiment", {})
        data_cfg = cfg["data"]
        self.loso_root = Path(exp_cfg.get("loso_root", data_cfg.get("loso_root", data_cfg["root"])))
        self.output_root = Path(cfg["project"]["output_dir"])
        self.folds = exp_cfg.get("folds")

    def _fold_dirs(self) -> list[Path]:
        if self.folds:
            folds = self.folds if isinstance(self.folds, (list, tuple)) else [self.folds]
            fold_dirs = [self.loso_root / f"loso_subject_{int(fold):02d}" for fold in folds]
        else:
            fold_dirs = sorted(self.loso_root.glob("loso_subject_*"))
        if not fold_dirs:
            raise FileNotFoundError(f"No LOSO fold directories found under {self.loso_root}")
        return fold_dirs

    def _fold_cfg(self, fold_dir: Path) -> dict:
        cfg = copy.deepcopy(self.cfg)
        cfg["project"]["name"] = f"{self.cfg['project']['name']}_{fold_dir.name}"
        cfg["project"]["output_dir"] = str(self.output_root / fold_dir.name)
        cfg["data"]["root"] = str(fold_dir)
        cfg.setdefault("experiment", {})["mode"] = self._inner_mode()
        return cfg

    def _inner_mode(self) -> str:
        exp_cfg = self.cfg.get("experiment", {})
        explicit = exp_cfg.get("inner_mode") or exp_cfg.get("method")
        if explicit:
            return str(explicit).lower()
        model_name = str(self.cfg.get("model", {}).get("name", "")).lower()
        if model_name.startswith("supcon"):
            return "supcon"
        if model_name.startswith("simclr"):
            return "simclr"
        return "ordinary"

    def _experiment_cls(self):
        mode = self._inner_mode()
        if mode in {"ordinary", "classification", "fog_classification"}:
            return ClassificationExperiment
        if mode in {"supcon", "supcon_pretrain", "supcon_finetune"}:
            return SupConExperiment
        if mode in {"simclr", "simclr_pretrain", "simclr_finetune"}:
            return SimCLRExperiment
        raise ValueError(f"Unsupported LOSO inner mode: {mode}")

    def run(self):
        if is_main_process(self.cfg):
            self.output_root.mkdir(parents=True, exist_ok=True)
        barrier(self.cfg)
        rows = []
        for fold_dir in self._fold_dirs():
            fold_name = fold_dir.name
            test_subject = fold_name.rsplit("_", 1)[-1]
            if is_main_process(self.cfg):
                print(f"\n===== {fold_name}: test subject {test_subject} =====")
            exp = self._experiment_cls()(self._fold_cfg(fold_dir))
            metrics = exp.run()
            if not is_main_process(self.cfg):
                continue

            ckpt = load_checkpoint(exp.best_checkpoint(), exp.build_model(), map_location="cpu")

            row = {
                "fold": fold_name,
                "test_subject": test_subject,
                "checkpoint": str(exp.best_checkpoint()),
                "best_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
                "best_val_f1_macro": (ckpt.get("metrics", {}) or {}).get("f1_macro") if isinstance(ckpt, dict) else None,
            }
            row.update({f"test_{key}": value for key, value in scalar_metrics(metrics).items()})
            fold_output_dir = self.output_root / fold_name
            row.update(_read_per_class_metrics(fold_output_dir, int(self.cfg["model"]["num_classes"])))
            confusion_matrix = _read_confusion_matrix(fold_output_dir / "confusion_matrix_test.csv")
            if confusion_matrix is not None:
                row["test_confusion_matrix"] = json.dumps(confusion_matrix, separators=(",", ":"))
            rows.append(row)
            write_csv(self.output_root / "loso_summary.csv", rows)

        if not is_main_process(self.cfg):
            barrier(self.cfg)
            return {}

        metric_keys = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if (key.startswith("test_") or key == "best_val_f1_macro")
                and isinstance(value, (int, float))
            }
        )
        confusion_matrices = [
            json.loads(row["test_confusion_matrix"])
            for row in rows
            if row.get("test_confusion_matrix")
        ]
        summary = {
            "num_folds": len(rows),
            "folds": rows,
            "aggregate": summarize(rows, metric_keys),
        }
        confusion_matrix_sum = _sum_matrices(confusion_matrices)
        if confusion_matrix_sum is not None:
            summary["confusion_matrix_test_sum"] = confusion_matrix_sum
        save_json(summary, self.output_root / "loso_summary.json")
        write_csv(self.output_root / "loso_summary.csv", rows)

        print("\n===== LOSO aggregate =====")
        primary_keys = [
            "test_f1_macro",
            "test_recall_macro",
            "test_pr_auc_macro",
            "test_pre_fog_recall",
            "test_pre_fog_f1",
            "test_fog_recall",
            "test_fog_f1",
            "test_accuracy",
            "test_balanced_accuracy",
        ]
        for key in primary_keys:
            stats = summary["aggregate"].get(key, {})
            if stats:
                print(f"{key}: mean={stats.get('mean')} std={stats.get('std')}")
        barrier(self.cfg)
        return summary
