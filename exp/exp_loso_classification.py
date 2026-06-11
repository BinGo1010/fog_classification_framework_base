from __future__ import annotations

import copy
from pathlib import Path

from exp.exp_fog_classification import ClassificationExperiment
from exp.exp_simclr import SimCLRExperiment
from exp.exp_supcon import SupConExperiment
from exp.exp_utils import scalar_metrics, summarize, write_csv
from utils.io import load_checkpoint, save_json
from utils.distributed import barrier, is_main_process


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
            rows.append(row)
            write_csv(self.output_root / "loso_summary.csv", rows)

        if not is_main_process(self.cfg):
            barrier(self.cfg)
            return {}

        metric_keys = [key for key in rows[0] if key.startswith("test_") or key == "best_val_f1_macro"]
        summary = {
            "num_folds": len(rows),
            "folds": rows,
            "aggregate": summarize(rows, metric_keys),
        }
        save_json(summary, self.output_root / "loso_summary.json")
        write_csv(self.output_root / "loso_summary.csv", rows)

        print("\n===== LOSO aggregate =====")
        for key in ["test_f1_macro", "test_balanced_accuracy", "test_accuracy", "test_roc_auc", "test_pr_auc"]:
            stats = summary["aggregate"].get(key, {})
            print(f"{key}: mean={stats.get('mean')} std={stats.get('std')}")
        barrier(self.cfg)
        return summary
