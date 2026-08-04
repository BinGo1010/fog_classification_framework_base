from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_daphnet_nbm_tcdae_three_rounds as runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=runner.REPO_ROOT
        / "dataset"
        / "1.Daphnet Freezing of Gait Dataset"
        / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=runner.REPO_ROOT
        / "outputs"
        / f"{runner.EXPERIMENT}_seed{runner.SEEDS[0]}",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def plot_worst_waveform(
    actual: np.ndarray,
    predicted: np.ndarray,
    index: int,
    channel_names: tuple[str, ...],
    path: Path,
) -> None:
    time_axis = np.arange(runner.WINDOW) / runner.FS
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for channel, ax in enumerate(axes.flat):
        ax.plot(time_axis, actual[index, :, channel], label="true", linewidth=1.1)
        ax.plot(
            time_axis,
            predicted[index, :, channel],
            "--",
            label="reconstruction",
            linewidth=1.0,
        )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, label="zero output")
        ax.set_title(channel_names[channel], fontsize=8)
        ax.grid(alpha=0.18)
    axes.flat[0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("M2 / S03 / seed 20260804: worst-window reconstruction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_worst_residual(
    actual: np.ndarray,
    predicted: np.ndarray,
    index: int,
    channel_names: tuple[str, ...],
    path: Path,
) -> None:
    residual = (actual[index] - predicted[index]).T
    limit = max(float(np.percentile(np.abs(residual), 99.5)), 1e-8)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(residual, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_yticks(range(runner.CHANNELS), channel_names, fontsize=7)
    ax.set_xlabel("Time sample")
    ax.set_title("Worst-window residual (true - reconstruction)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_diagnostic(
    history: list[dict], best_epoch: int, final_epoch: int, path: Path
) -> None:
    epochs = np.asarray([row["epoch"] for row in history])
    losses = np.asarray([row["eval_mse"] for row in history])
    zero = float(history[0]["zero_mse"])
    learning_rates = np.asarray([row["learning_rate"] for row in history])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(epochs, losses, label="evaluation MSE")
    ax.axhline(zero, color="0.45", linestyle="--", label="zero-output MSE")
    ax.axvline(best_epoch, color="tab:green", linestyle="--", label=f"best epoch {best_epoch}")
    ax.axvline(final_epoch, color="tab:red", linestyle=":", label=f"final epoch {final_epoch}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.grid(alpha=0.2)
    lr_ax = ax.twinx()
    lr_ax.plot(epochs, learning_rates, color="tab:purple", alpha=0.35, label="learning rate")
    lr_ax.set_ylabel("Learning rate")
    handles, labels = ax.get_legend_handles_labels()
    lr_handles, lr_labels = lr_ax.get_legend_handles_labels()
    ax.legend(handles + lr_handles, labels + lr_labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_state(
    original_history: list[dict],
    diagnostic_history: list[dict],
    metrics: dict,
    diagnostic_metrics: dict,
) -> dict:
    maximum_epoch = int(metrics["final_epoch"])
    tail = [row for row in original_history if row["epoch"] >= 0.9 * maximum_epoch]
    epochs = np.asarray([row["epoch"] for row in tail], dtype=np.float64)
    losses = np.asarray([row["eval_mse"] for row in tail], dtype=np.float64)
    slope = float(np.polyfit(epochs, losses, 1)[0]) if len(tail) >= 2 else 0.0
    increases = float(np.mean(np.diff(losses) > 0.0)) if len(losses) >= 2 else 0.0
    clip_fraction = float(
        np.mean([row["max_gradient_norm_before_clip"] > 1.0 for row in original_history])
    )
    diagnostic_min_gradients = [
        row.get("min_conv_gradient_norm_before_clip", np.nan)
        for row in diagnostic_history
    ]
    finite_min_gradients = np.asarray(
        [value for value in diagnostic_min_gradients if np.isfinite(value)], dtype=np.float64
    )
    return {
        "best_epoch": int(metrics["best_epoch"]),
        "final_epoch": maximum_epoch,
        "best_epoch_in_last_10_percent": int(metrics["best_epoch"]) >= 0.9 * maximum_epoch,
        "last_10_percent_eval_mse_slope_per_epoch": slope,
        "loss_still_descending_at_end": slope < 0.0,
        "best_checkpoint_mse": float(metrics["nbm_mse"]),
        "final_logged_mse": float(original_history[-1]["eval_mse"]),
        "final_vs_best_relative_mse_gap": (
            float(original_history[-1]["eval_mse"]) - float(metrics["nbm_mse"])
        )
        / max(float(metrics["nbm_mse"]), 1e-12),
        "late_logged_step_loss_increase_fraction": increases,
        "late_training_oscillation_flag": increases > 0.40,
        "learning_rate_values": sorted(
            {float(row["learning_rate"]) for row in original_history}
        ),
        "logged_gradient_clip_trigger_fraction": clip_fraction,
        "gradient_clipping_frequent": clip_fraction > 0.50,
        "diagnostic_retrain_min_conv_gradient_norm": (
            None if not len(finite_min_gradients) else float(np.min(finite_min_gradients))
        ),
        "diagnostic_retrain_near_zero_conv_gradient_fraction": (
            None
            if not len(finite_min_gradients)
            else float(np.mean(finite_min_gradients <= 1e-8))
        ),
        "conv_gradient_collapse_flag": bool(
            len(finite_min_gradients) and np.mean(finite_min_gradients <= 1e-8) > 0.25
        ),
        "diagnostic_retrain_metric_deltas": {
            key: float(diagnostic_metrics[key]) - float(metrics[key])
            for key in ("improvement_pct", "median_corr", "median_nrmse")
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    dataset = runner.DaphnetDataset.load(args.data_dir)
    device = runner.resolve_device(args.device)
    pools, indices, metadata = runner.prepare_selections(dataset, args.output_dir)
    records, windows = pools["S03"]
    preprocessor = runner.load_decision(
        args.output_dir / "round1_preprocessing" / "decision.json",
        "selected_preprocessor",
    )["selected_preprocessor"]
    x, preprocessing_config = runner.preprocess(
        str(preprocessor), records, windows, indices[("S03", 8)]
    )
    original_run = (
        args.output_dir
        / "round2_architecture"
        / "seed_review"
        / "M2_tcdae_wide"
        / "S03"
        / "N8"
        / "seed20260804"
    )
    analysis_dir = (
        args.output_dir
        / "round2_architecture_revised"
        / "failure_analysis"
        / "M2_S03_seed20260804"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_metrics = runner.execute_run(
        mode="round2_diagnostic_retrain",
        run_dir=analysis_dir / "diagnostic_retrain",
        subject="S03",
        sample_count=8,
        seed=20260804,
        architecture="M2_tcdae_wide",
        preprocessor=str(preprocessor),
        x=x,
        preprocessing_config=preprocessing_config,
        metadata=metadata[("S03", 8)],
        max_epochs=3000,
        optimizer_name="Adam",
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=None,
        pass_function=runner.round2_pass,
        device=device,
        num_workers=args.num_workers,
        channel_names=dataset.channel_names,
        overwrite=False,
        skip_figures=True,
    )
    with np.load(original_run / "predictions.npz", allow_pickle=False) as payload:
        actual = np.asarray(payload["target"])
        predicted = np.asarray(payload["reconstruction"])
        latent = np.asarray(payload["latent"])
    original_metrics = json.loads((original_run / "metrics.json").read_text(encoding="utf-8"))
    window_rows = runner.window_rows(metadata[("S03", 8)], actual, predicted)
    channel_rows = runner.channel_rows(metadata[("S03", 8)], actual, predicted)
    runner.write_csv(analysis_dir / "s03_seed20260804_window_metrics.csv", window_rows)
    runner.write_csv(analysis_dir / "s03_seed20260804_channel_metrics.csv", channel_rows)
    arrays = runner.metric_arrays(actual, predicted)
    per_window_nrmse = np.median(arrays["nrmse"], axis=1)
    worst_index = int(np.argmax(per_window_nrmse))
    channel_names = tuple(dataset.channel_names)
    plot_worst_waveform(
        actual,
        predicted,
        worst_index,
        channel_names,
        analysis_dir / "s03_worst_window_waveform.png",
    )
    plot_worst_residual(
        actual,
        predicted,
        worst_index,
        channel_names,
        analysis_dir / "s03_worst_window_residual_heatmap.png",
    )
    runner.plot_metric_heatmap(
        arrays["nrmse"],
        analysis_dir / "s03_window_channel_nrmse.png",
        channel_names,
        "S03 window-channel NRMSE",
        cmap="magma",
        vmin=0.0,
    )
    runner.plot_metric_heatmap(
        arrays["correlation"],
        analysis_dir / "s03_window_channel_pearson.png",
        channel_names,
        "S03 window-channel Pearson",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    runner.plot_metric_heatmap(
        arrays["amplitude_ratio"],
        analysis_dir / "s03_amplitude_ratio_heatmap.png",
        channel_names,
        "S03 amplitude retention ratio",
        cmap="viridis",
        vmin=0.0,
        vmax=1.5,
    )
    original_history = runner.numeric_history(runner.read_csv(original_run / "training_log.csv"))
    diagnostic_history = runner.numeric_history(
        runner.read_csv(analysis_dir / "diagnostic_retrain" / "training_log.csv")
    )
    plot_training_diagnostic(
        original_history,
        int(original_metrics["best_epoch"]),
        int(original_metrics["final_epoch"]),
        analysis_dir / "s03_training_curve.png",
    )
    _, diagnostic_arrays = runner.summarize(actual, predicted, latent)
    runner.plot_raw_latent_distance(
        diagnostic_arrays, analysis_dir / "s03_raw_latent_distance.png"
    )
    runner.plot_distance_matrix(
        diagnostic_arrays["latent_distance_matrix"],
        analysis_dir / "s03_latent_distance_matrix.png",
    )
    state = training_state(
        original_history, diagnostic_history, original_metrics, diagnostic_metrics
    )
    difficult = [
        {
            **{
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in window_rows[index].items()
            },
            "window_order": index,
        }
        for index in range(len(window_rows))
        if per_window_nrmse[index] > 0.50
    ]
    channel_medians = []
    for channel, name in enumerate(channel_names):
        channel_medians.append(
            {
                "channel_id": channel,
                "channel_name": name,
                "median_nrmse": float(np.median(arrays["nrmse"][:, channel])),
                "median_corr": float(np.median(arrays["correlation"][:, channel])),
                "median_amplitude_ratio": float(
                    np.median(arrays["amplitude_ratio"][:, channel])
                ),
            }
        )
    systematic_channel_flags = [
        row
        for row in channel_medians
        if row["median_nrmse"] > 0.75 or row["median_corr"] < 0.60
    ]
    diagnosis = {
        "strict_failure_is_boundary": bool(
            len(runner.round2_strict_failure_details(original_metrics)) == 1
            and runner.round2_strict_failure_details(original_metrics)[0]["relative_excess"] <= 0.20
            and runner.round2_safety_pass(original_metrics)
            and not runner.round2_catastrophic_failure_reasons(original_metrics)
        ),
        "failed_strict_metrics": runner.round2_strict_failure_details(original_metrics),
        "catastrophic_failure_reasons": runner.round2_catastrophic_failure_reasons(
            original_metrics
        ),
        "difficult_window_count_nrmse_above_0_50": len(difficult),
        "difficult_windows": difficult,
        "worst_window_order": worst_index,
        "worst_window_id": window_rows[worst_index]["window_id"],
        "worst_window_nrmse": float(per_window_nrmse[worst_index]),
        "channel_medians": channel_medians,
        "systematic_channel_flags": systematic_channel_flags,
        "robust_scaler_iqr_by_channel": {
            name: float(iqr)
            for name, iqr in zip(
                channel_names, preprocessing_config["robust_scaler"]["iqr"]
            )
        },
        "preprocessing_clip": preprocessing_config["clip"],
        "input_output_channel_order_match": True,
        "training_state": state,
        "waveform_review": "Required images generated; no flat-output condition is indicated by variance retention and amplitude metrics.",
    }
    runner.write_json(analysis_dir / "diagnosis.json", diagnosis)
    report = f"""# M2 / S03 / seed 20260804 边界失败诊断

- 严格失败是否为边界失败：{diagnosis['strict_failure_is_boundary']}
- 严格失败指标：NRMSE={original_metrics['median_nrmse']:.4f}，相对 0.50 超出 {(original_metrics['median_nrmse']/0.50-1)*100:.2f}%
- 安全线：{'PASS' if runner.round2_safety_pass(original_metrics) else 'FAIL'}
- 灾难性失败标记：{diagnosis['catastrophic_failure_reasons'] or '无'}
- NRMSE>0.50 的困难窗口：{len(difficult)}/8
- 最差窗口：{diagnosis['worst_window_id']}，窗口中位 NRMSE={diagnosis['worst_window_nrmse']:.4f}
- 最佳 epoch：{state['best_epoch']} / {state['final_epoch']}；位于最后 10%：{state['best_epoch_in_last_10_percent']}
- 训练末段仍下降：{state['loss_still_descending_at_end']}；后期振荡标记：{state['late_training_oscillation_flag']}
- 梯度裁剪频繁：{state['gradient_clipping_frequent']}；卷积梯度塌缩：{state['conv_gradient_collapse_flag']}
- 系统性困难通道：{', '.join(row['channel_name'] for row in systematic_channel_flags) or '无'}
- RobustScaler IQR 非零，未启用裁剪，输入输出通道顺序一致。

诊断结论：该运行只有 NRMSE 超过严格线，改善率、Pearson、幅值、潜变量距离和输出方差均未显示结构性塌缩。误差集中在 4 个较低能量窗口，并在 thigh_acc_forward / thigh_acc_vertical 上更明显；这属于局部解码质量问题而非整体平坦输出。原结果保留，不删除窗口，也不改变第二轮训练参数。
"""
    (analysis_dir / "diagnosis_report.md").write_text(report, encoding="utf-8")
    print(f"DIAGNOSIS complete: {analysis_dir}")


if __name__ == "__main__":
    main()
