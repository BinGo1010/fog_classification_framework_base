"""Aggregate subject-sharded Daphnet NBM E0--E3 runs.

Training is independent within subject/seed, but residual-score selection and
E2/E3 gates are population-level operations.  This command rehydrates each
GPU shard's saved predictions and produces the only authoritative global gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "scripts"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_daphnet_nbm_e0_e3_a5_50 as study  # noqa: E402


def parse_args() -> argparse.Namespace:
    dataset = ROOT / "dataset" / "1.Daphnet Freezing of Gait Dataset" / "processed_A5_50"
    reference = (
        ROOT
        / "outputs"
        / "daphnet_nbm_routeA_A5_50_manifest_full_v1"
        / "routeA_A5_50_manifest_full"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=dataset)
    parser.add_argument("--shard-parent", type=Path)
    parser.add_argument("--shard-roots", type=Path, nargs="*")
    parser.add_argument("--shard-glob", default="gpu*")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "daphnet_nbm_E0_E3_A5_50_7gpu_v1",
    )
    parser.add_argument("--phase", choices=("e2", "e3"), required=True)
    parser.add_argument("--seeds", default=",".join(map(str, study.SEEDS)))
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--include-e2-p16", action="store_true")
    parser.add_argument("--include-e3b", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--reference-a5-root", type=Path, default=reference)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def discover_shards(args: argparse.Namespace) -> list[Path]:
    roots = [path.resolve() for path in (args.shard_roots or [])]
    if args.shard_parent:
        roots.extend(
            path.resolve()
            for path in sorted(args.shard_parent.resolve().glob(args.shard_glob))
            if path.is_dir()
        )
    roots = list(dict.fromkeys(roots))
    if not roots:
        raise ValueError("no shard roots found; supply --shard-parent or --shard-roots")
    return roots


def shard_subjects(
    shard_roots: Sequence[Path], seeds: Sequence[int], *, allow_partial: bool
) -> tuple[dict[Path, tuple[str, ...]], tuple[str, ...], list[dict[str, Any]]]:
    mapping: dict[Path, tuple[str, ...]] = {}
    owner: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for shard in shard_roots:
        protocol_path = shard / "protocol" / "frozen_protocol.json"
        if not protocol_path.exists():
            raise FileNotFoundError(f"missing shard protocol: {protocol_path}")
        protocol = study.read_json(protocol_path)
        shard_seeds = tuple(int(value) for value in protocol.get("seeds", []))
        if shard_seeds != tuple(seeds):
            raise ValueError(
                f"seed mismatch in {protocol_path}: expected {tuple(seeds)}, got {shard_seeds}"
            )
        subjects = tuple(str(value) for value in protocol.get("subjects", []))
        if not subjects:
            raise ValueError(f"empty subject shard: {protocol_path}")
        for subject in subjects:
            if subject in owner:
                raise ValueError(f"subject {subject} appears in both {owner[subject]} and {shard}")
            owner[subject] = shard
            rows.append(
                {
                    "subject_id": subject,
                    "shard_root": str(shard),
                    "seeds": ",".join(map(str, seeds)),
                }
            )
        mapping[shard] = subjects
    all_subjects = tuple(subject for subject in study.ALL_SUBJECTS if subject in owner)
    missing = sorted(set(study.ALL_SUBJECTS) - set(all_subjects))
    extra = sorted(set(owner) - set(study.ALL_SUBJECTS))
    if extra:
        raise ValueError(f"unknown subjects in shards: {extra}")
    if missing and not allow_partial:
        raise ValueError(f"incomplete global protocol; missing subjects: {missing}")
    return mapping, all_subjects, rows


def load_stage(
    stage_dir: str,
    expected_stage: str,
    shard_map: dict[Path, tuple[str, ...]],
    bundles: dict[str, study.SubjectBundle],
    seeds: Sequence[int],
) -> dict[tuple[str, int], study.RunOutputs]:
    outputs: dict[tuple[str, int], study.RunOutputs] = {}
    for shard, subjects in shard_map.items():
        subset = {subject: bundles[subject] for subject in subjects}
        current = study.load_saved_runs(
            shard / stage_dir,
            subset,
            seeds,
            expected_stage=expected_stage,
        )
        duplicates = sorted(set(outputs) & set(current))
        if duplicates:
            raise ValueError(f"duplicate saved runs: {duplicates}")
        outputs.update(current)
    expected = len(bundles) * len(seeds)
    if len(outputs) != expected:
        raise ValueError(f"{expected_stage} has {len(outputs)} runs; expected {expected}")
    return outputs


def write_training_summary(stage_root: Path, runs: dict[tuple[str, int], study.RunOutputs]) -> None:
    rows = []
    for (subject, seed), run in sorted(runs.items()):
        rows.append(
            {
                "subject_id": subject,
                "seed": seed,
                "source_run_dir": str(run.run_dir),
                **run.training,
            }
        )
    study.write_csv(stage_root / "training_summary.csv", rows)


def build_protocol(
    args: argparse.Namespace,
    subjects: Sequence[str],
    seeds: Sequence[int],
    manifest_path: Path,
    support_audit: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    protocol_args = Namespace(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        smoke=args.allow_partial,
    )
    protocol = study.protocol_payload(
        protocol_args, subjects, seeds, manifest_path, support_audit
    )
    protocol["execution"] = {
        "mode": "subject_sharded_multi_gpu",
        "authoritative_gate_scope": "all shard subjects",
        "shard_local_gates_authoritative": False,
    }
    return protocol


def main() -> None:
    args = parse_args()
    seeds = study.parse_seeds(args.seeds)
    shard_roots = discover_shards(args)
    shard_map, subjects, shard_rows = shard_subjects(
        shard_roots, seeds, allow_partial=args.allow_partial
    )
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    bundles, _, channel_names = study.load_data(
        data_dir, subjects, max_windows_per_role=0
    )
    support, support_audit = study.common_support(bundles)
    manifest_path = study.manifest_a5.resolve_a5_artifact(data_dir, "a5_window_manifest.csv")
    protocol = build_protocol(args, subjects, seeds, manifest_path, support_audit)
    study.write_json(root / "protocol" / "frozen_protocol.json", protocol)
    study.write_csv(root / "protocol" / "shard_index.csv", shard_rows)
    study.write_csv(root / "protocol" / "E3A_common_support_audit.csv", support_audit)
    study.write_json(
        root / "protocol" / "train_only_scalers.json",
        {subject: item.scaler for subject, item in bundles.items()},
    )
    split_audit: list[dict[str, Any]] = []
    for subject, item in bundles.items():
        for role in study.ROLES:
            split_audit.append(
                {
                    "subject_id": subject,
                    "subject_scope": item.scope,
                    "a5_role": role,
                    "windows_loaded": len(item.role_rows[role]),
                    "E3A_common_support_windows": len(support[(subject, role)]),
                }
            )
    study.write_csv(root / "protocol" / "split_audit.csv", split_audit)

    results: dict[str, dict[str, Any]] = {}
    e0_runs = load_stage("E0", "E0", shard_map, bundles, seeds)
    write_training_summary(root / "E0", e0_runs)
    e0_full = study.evaluate_stage(
        "E0",
        e0_runs,
        root / "E0",
        calibration="C0",
        support=None,
        weight_step=args.weight_step,
        channel_names=channel_names,
        suffix="_a5_full",
    )
    study.compare_e0_reference(
        root,
        e0_full,
        args.reference_a5_root.resolve(),
        subjects=subjects,
        seeds=seeds,
    )
    results["E0"] = study.evaluate_stage(
        "E0",
        e0_runs,
        root / "E0",
        calibration="C0",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    study.write_stage_contract_files(
        root / "E0", "E0", results["E0"], protocol=protocol, split_audit=split_audit
    )

    results["E1"] = study.evaluate_stage(
        "E1",
        e0_runs,
        root / "E1",
        calibration="C1",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    study.write_stage_contract_files(
        root / "E1", "E1", results["E1"], protocol=protocol, split_audit=split_audit
    )
    study.write_csv(
        root / "E1" / "e0_e1_paired_metrics.csv",
        study.paired_metric_table(results["E0"], results["E1"], "E0", "E1"),
    )
    study.write_json(root / "E1" / "E1_gate.json", study.e1_gate(results["E0"], results["E1"]))

    e2_runs = load_stage("E2", "E2", shard_map, bundles, seeds)
    write_training_summary(root / "E2", e2_runs)
    results["E2"] = study.evaluate_stage(
        "E2",
        e2_runs,
        root / "E2",
        calibration="C1",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    study.write_stage_contract_files(
        root / "E2", "E2", results["E2"], protocol=protocol, split_audit=split_audit
    )
    study.write_csv(
        root / "E2" / "e1_e2_paired_metrics.csv",
        study.paired_metric_table(results["E1"], results["E2"], "E1", "E2"),
    )
    e2_decision = study.e2_gate(results["E1"], results["E2"])
    study.write_json(root / "E2" / "E2_gate.json", e2_decision)
    study.reconstruction_gap(
        results["E2"], results["E1"], root / "E2" / "fog_nonfog_reconstruction_gap.csv"
    )
    study.write_csv(
        root / "E2" / "probe_classifier_results.csv",
        [{
            "status": "NE",
            "reason": "no preregistered A5_50-compatible frozen Raw-TCN checkpoint was supplied",
            "test_used_for_probe_selection": False,
        }],
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for aggregation profile, but CUDA is unavailable")
    profiles = []
    for name, model, shape in (
        ("E0_M3", study.a1b.ContextM3(study.WINDOW), (1, study.CHANNELS, study.WINDOW)),
        ("E2_P24", study.TrueBottleneckAE("P24"), (1, study.CHANNELS, study.WINDOW)),
    ):
        profiles.append({"model": name, **study.profile_model(model, shape, device=device)})
    study.write_csv(root / "E2" / "parameter_comparison.csv", profiles)
    study.write_json(
        root / "E2" / "architecture_summary.json",
        study.TrueBottleneckAE("P24").architecture_config(),
    )
    (root / "E2" / "architecture_summary.md").write_text(
        study.architecture_markdown(
            "E2-P24", study.TrueBottleneckAE("P24").architecture_config(), profiles[1]
        ),
        encoding="utf-8",
    )

    if args.include_e2_p16:
        p16_runs = load_stage("E2_P16", "E2_P16", shard_map, bundles, seeds)
        write_training_summary(root / "E2_P16", p16_runs)
        results["E2_P16"] = study.evaluate_stage(
            "E2_P16",
            p16_runs,
            root / "E2_P16",
            calibration="C1",
            support=support,
            weight_step=args.weight_step,
            channel_names=channel_names,
        )
        study.write_stage_contract_files(
            root / "E2_P16",
            "E2_P16",
            results["E2_P16"],
            protocol=protocol,
            split_audit=split_audit,
        )

    capacity = 24 if e2_decision["status"] == "PASS" else 48
    capacity_payload = {
        "latent_channels": capacity,
        "runner_argument": "p24" if capacity == 24 else "m3",
        "reason": (
            "global E2 P24 gate passed before E3"
            if capacity == 24
            else "global E2 P24 gate did not pass; retain M3-equivalent capacity"
        ),
        "gate_scope": list(subjects),
        "shard_local_gates_used_for_choice": False,
        "E3_test_used_for_choice": False,
    }
    study.write_json(root / "E3_capacity_decision.json", capacity_payload)
    if args.phase == "e2":
        study.comparison_outputs(root, results, bootstrap_samples=args.bootstrap_samples)
        print(
            f"GLOBAL E2 COMPLETE capacity={capacity_payload['runner_argument']} results={root}",
            flush=True,
        )
        return

    e3a_runs = load_stage("E3", "E3A", shard_map, bundles, seeds)
    expected_token = f"C{capacity}"
    inconsistent = [
        f"{subject}/seed{seed}:{run.model_name}"
        for (subject, seed), run in e3a_runs.items()
        if expected_token not in run.model_name
    ]
    if inconsistent:
        raise ValueError(
            "E3 capacity does not match global E2 decision; first mismatches: "
            + ", ".join(inconsistent[:5])
        )
    write_training_summary(root / "E3", e3a_runs)
    results["E3A"] = study.evaluate_stage(
        "E3A",
        e3a_runs,
        root / "E3",
        calibration="C1",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    study.write_stage_contract_files(
        root / "E3", "E3A", results["E3A"], protocol=protocol, split_audit=split_audit
    )
    context_audit: list[dict[str, Any]] = []
    for item in bundles.values():
        for role in study.ROLES:
            _, _, _, audit = study.context_target_arrays(item, role, mode="E3A")
            context_audit.extend(audit)
    study.write_csv(root / "E3" / "context_target_manifest.csv", context_audit)
    e3_model = study.HistoryPredictor(capacity, study.CHANNELS)
    e3_profile = study.profile_model(
        e3_model, (1, study.CHANNELS, 256), device=device
    )
    study.write_json(root / "E3" / "architecture_summary.json", e3_model.architecture_config())
    (root / "E3" / "architecture_summary.md").write_text(
        study.architecture_markdown("E3-A", e3_model.architecture_config(), e3_profile),
        encoding="utf-8",
    )
    study.write_csv(
        root / "E3" / "fog_normalization_probe.csv",
        [{
            "status": "NE",
            "reason": "no preregistered A5_50-compatible frozen Raw-TCN checkpoint was supplied",
            "test_used_for_probe_selection": False,
        }],
    )
    lowpass = study.evaluate_stage(
        "LOWPASS8",
        study.lowpass_runs(e3a_runs),
        root / "E3" / "lowpass_baseline",
        calibration="C1",
        support=support,
        weight_step=args.weight_step,
        channel_names=channel_names,
    )
    study.write_csv(root / "E3" / "lowpass_baseline_metrics.csv", lowpass["test_rows"])
    baseline = results["E2"] if e2_decision["status"] == "PASS" else results["E1"]
    study.write_json(
        root / "E3" / "E3_gate.json",
        study.e3_gate(baseline, results["E3A"], lowpass, probe_available=False),
    )

    if args.include_e3b:
        e3b_runs = load_stage("E3B", "E3B", shard_map, bundles, seeds)
        write_training_summary(root / "E3B", e3b_runs)
        results["E3B"] = study.evaluate_stage(
            "E3B",
            e3b_runs,
            root / "E3B",
            calibration="C1",
            support=support,
            weight_step=args.weight_step,
            channel_names=channel_names,
        )
        study.write_stage_contract_files(
            root / "E3B", "E3B", results["E3B"], protocol=protocol, split_audit=split_audit
        )

    study.comparison_outputs(root, results, bootstrap_samples=args.bootstrap_samples)
    full_protocol = set(study.FORMAL_SUBJECTS).issubset(subjects) and set(study.SEEDS).issubset(seeds)
    final_gate = study.final_r5_eligibility(results, full_protocol=full_protocol)
    study.write_json(root / "R5_TCN_eligibility_gate.json", final_gate)
    study.render_final_report(root, final_gate)
    print(f"GLOBAL E0-E3 COMPLETE results={root}", flush=True)


if __name__ == "__main__":
    main()
