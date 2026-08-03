"""
Post-run analysis for the closed-loop radar-to-control experiment.

Loads the metrics CSV produced by closed_loop_runner.py and generates:
  - Summary table (false-brake count, mean TTC, min following distance, latency)
  - Inference latency histogram
  - Following distance over time (detected vs oracle)
  - False-brake event timeline
  - Time-to-collision (TTC) distribution

Usage:
    python3 benchmark_closed_loop.py --csv ./closed_loop_output/closed_loop_metrics.csv
    python3 benchmark_closed_loop.py --csv ./closed_loop_output/closed_loop_metrics.csv \\
        --out-dir ./closed_loop_output/plots
"""

import os
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# TTC is undefined when closing velocity ≤ 0 (not approaching)
TTC_MAX_S = 30.0   # clip TTC at this value for plotting
EGO_MIN_SPEED_MS = 0.5   # below this speed, TTC is meaningless


def load_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Coerce numeric columns (some are "nan" strings)
    numeric_cols = [
        "inference_ms", "n_detections",
        "detected_range_m", "detected_vel_ms",
        "gt_range_m", "gt_closing_vel_ms",
        "throttle", "brake", "steer",
        "ego_speed_ms",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_ttc(range_m: pd.Series, closing_vel_ms: pd.Series) -> pd.Series:
    """TTC = range / closing_vel (only when closing_vel > 0)."""
    ttc = range_m / closing_vel_ms.clip(lower=1e-3)
    ttc[closing_vel_ms <= 0] = np.nan
    ttc[range_m.isna()] = np.nan
    return ttc.clip(upper=TTC_MAX_S)


def print_summary(df: pd.DataFrame):
    hot = df[df["cold_start"] == 0].copy()

    n_frames = len(hot)
    false_brakes = int(hot["false_brake"].sum())
    pct_false = 100.0 * false_brakes / n_frames if n_frames > 0 else 0.0

    latency = hot["inference_ms"].dropna()
    mean_lat = latency.mean()
    p95_lat = latency.quantile(0.95)
    max_lat = latency.max()
    budget_ms = 1000.0 / 20.0  # 50 ms at 20 FPS
    pct_over = 100.0 * (latency > budget_ms).mean()

    gt_range = hot["gt_range_m"].dropna()
    min_gt_dist = gt_range.min()
    mean_gt_dist = gt_range.mean()

    det_range = hot["detected_range_m"].dropna()

    # TTC from detected range + detected closing velocity
    hot["ttc_detected"] = compute_ttc(hot["detected_range_m"], hot["detected_vel_ms"])
    hot["ttc_oracle"] = compute_ttc(hot["gt_range_m"], hot["gt_closing_vel_ms"])
    mean_ttc_det = hot["ttc_detected"].dropna().mean()
    min_ttc_det = hot["ttc_detected"].dropna().min()

    print("\n" + "=" * 60)
    print("  CLOSED-LOOP BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Total frames (post-cold-start): {n_frames}")
    print(f"  False-brake events            : {false_brakes}  ({pct_false:.1f}%)")
    print()
    print(f"  Inference latency (ms):")
    print(f"    Mean    : {mean_lat:.1f}")
    print(f"    P95     : {p95_lat:.1f}")
    print(f"    Max     : {max_lat:.1f}")
    print(f"    Budget  : {budget_ms:.0f}  (20 Hz)")
    print(f"    >budget : {pct_over:.1f}%")
    print()
    print(f"  Following distance — oracle (m):")
    print(f"    Min     : {min_gt_dist:.2f}")
    print(f"    Mean    : {mean_gt_dist:.2f}")
    print()
    print(f"  Time-to-collision — detected (s):")
    if not math.isnan(mean_ttc_det):
        print(f"    Mean    : {mean_ttc_det:.2f}")
        print(f"    Min     : {min_ttc_det:.2f}")
    else:
        print(f"    (no approaching events detected)")
    print("=" * 60 + "\n")


def plot_all(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    hot = df[df["cold_start"] == 0].copy()
    hot["ttc_detected"] = compute_ttc(hot["detected_range_m"], hot["detected_vel_ms"])
    hot["ttc_oracle"] = compute_ttc(hot["gt_range_m"], hot["gt_closing_vel_ms"])
    frames = hot["frame"].values
    budget_ms = 1000.0 / 20.0

    # ---- Figure 1: inference latency histogram ----
    fig, ax = plt.subplots(figsize=(7, 4))
    lat = hot["inference_ms"].dropna()
    ax.hist(lat, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(budget_ms, color="red", linestyle="--", linewidth=1.5, label=f"20 Hz budget ({budget_ms:.0f} ms)")
    ax.axvline(lat.mean(), color="orange", linestyle="-", linewidth=1.5, label=f"Mean ({lat.mean():.1f} ms)")
    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Frame count")
    ax.set_title("Online Radar Detector Inference Latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_histogram.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved latency_histogram.png")

    # ---- Figure 2: following distance (detected vs oracle) over time ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, hot["gt_range_m"].values, color="green", linewidth=1.0,
            label="Oracle range (GT)", alpha=0.85)
    ax.plot(frames, hot["detected_range_m"].values, color="steelblue", linewidth=0.8,
            label="Detected range (model)", alpha=0.7)
    fb_frames = frames[hot["false_brake"].values.astype(bool)]
    if len(fb_frames) > 0:
        ax.scatter(fb_frames, np.full_like(fb_frames, 0.0, dtype=float),
                   marker="v", color="red", s=30, label="False brake event", zorder=5)
    ax.axhline(15.0, color="orange", linestyle="--", linewidth=1.0, label="Safe distance (15 m)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Range (m)")
    ax.set_title("Following Distance: Detected vs Oracle")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "following_distance.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved following_distance.png")

    # ---- Figure 3: brake output + false-brake timeline ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(frames, hot["brake"].values, color="crimson", linewidth=0.8, label="Brake command")
    axes[0].set_ylabel("Brake [0–1]")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].fill_between(frames, hot["false_brake"].values.astype(float),
                         step="mid", color="red", alpha=0.6, label="False brake")
    axes[1].set_ylabel("False brake")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylim(-0.1, 1.5)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Brake Commands and False-Brake Events")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "brake_timeline.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved brake_timeline.png")

    # ---- Figure 4: TTC distribution ----
    ttc_det = hot["ttc_detected"].dropna()
    ttc_ora = hot["ttc_oracle"].dropna()
    if len(ttc_det) > 0 or len(ttc_ora) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(ttc_ora) > 0:
            ax.hist(ttc_ora, bins=30, alpha=0.6, color="green", edgecolor="white",
                    label="Oracle TTC", density=True)
        if len(ttc_det) > 0:
            ax.hist(ttc_det, bins=30, alpha=0.6, color="steelblue", edgecolor="white",
                    label="Detected TTC", density=True)
        ax.set_xlabel("Time-to-Collision (s)")
        ax.set_ylabel("Density")
        ax.set_title("Time-to-Collision Distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "ttc_distribution.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved ttc_distribution.png")
    else:
        print("  [skip] TTC distribution — no approaching events found.")

    # ---- Figure 5: overview dashboard ----
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    lat = hot["inference_ms"].dropna()
    ax1.hist(lat, bins=30, color="steelblue", edgecolor="white")
    ax1.axvline(budget_ms, color="red", linestyle="--", label=f"Budget ({budget_ms:.0f} ms)")
    ax1.set_title("Inference Latency")
    ax1.set_xlabel("ms")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(frames, hot["gt_range_m"].values, color="green", label="Oracle", linewidth=0.8)
    ax2.plot(frames, hot["detected_range_m"].values, color="steelblue", label="Detected", linewidth=0.8, alpha=0.7)
    ax2.axhline(15.0, color="orange", linestyle="--", linewidth=1.0)
    ax2.set_title("Following Distance (m)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(frames, hot["brake"].values, color="crimson", linewidth=0.8)
    ax3.fill_between(frames, hot["false_brake"].values.astype(float), alpha=0.4, color="red",
                     step="mid", label="False brake")
    ax3.set_title("Brake Commands")
    ax3.set_xlabel("Frame")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if len(ttc_det) > 0:
        ax4.hist(ttc_det, bins=25, color="steelblue", alpha=0.7, density=True, edgecolor="white")
    ax4.set_title("TTC Distribution (detected)")
    ax4.set_xlabel("s")
    ax4.grid(alpha=0.3)

    fig.suptitle("Closed-Loop Radar Detector + ACC Controller — Benchmark Dashboard", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "dashboard.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved dashboard.png")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark analysis for closed-loop radar-to-control run")
    p.add_argument("--csv", required=True, help="Path to closed_loop_metrics.csv")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for plots (default: same dir as CSV)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(args.csv))

    print(f"Loading metrics from: {args.csv}")
    df = load_metrics(args.csv)
    print(f"  {len(df)} rows loaded ({df['cold_start'].sum()} cold-start frames excluded from analysis)")

    print_summary(df)

    print(f"\nGenerating plots → {out_dir}")
    plot_all(df, out_dir)

    print("Done.")
