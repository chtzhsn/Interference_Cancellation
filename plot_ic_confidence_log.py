"""
plot_ic_confidence_log.py

Generic plotting script for trajectory IC logs with confidence-bucket fields.

Usage from project root:
    python plot_ic_confidence_log.py --log logs/v3.9_v_only_conf_bucket.txt
    python plot_ic_confidence_log.py --log logs/v3.10_kv_conf_bucket.txt

Output:
    plots/<log_file_stem>/

Expected log fields:
    iter, loss, base_loss, ic_loss,
    delta_k_norm, delta_v_norm,
    gate_k_mean, gate_v_mean,
    base_top1, ic_top1,
    high_n, high_delta,
    med_n, med_delta,
    low_n, low_delta

Also parses eval lines:
    step N: train loss X, val loss Y
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


ITER_RE = re.compile(
    r"iter\s+(?P<iter>\d+):\s+"
    r"loss\s+(?P<loss>[-+]?\d*\.?\d+),\s+"
    r"base_loss\s+(?P<base_loss>[-+]?\d*\.?\d+),\s+"
    r"ic_loss\s+(?P<ic_loss>[-+]?\d*\.?\d+),\s+"
    r"delta_k_norm\s+(?P<delta_k_norm>[-+]?\d*\.?\d+),\s+"
    r"delta_v_norm\s+(?P<delta_v_norm>[-+]?\d*\.?\d+),\s+"
    r"gate_k_mean\s+(?P<gate_k_mean>[-+]?\d*\.?\d+),\s+"
    r"gate_v_mean\s+(?P<gate_v_mean>[-+]?\d*\.?\d+),\s+"
    r"base_top1\s+(?P<base_top1>[-+]?\d*\.?\d+),\s+"
    r"ic_top1\s+(?P<ic_top1>[-+]?\d*\.?\d+),\s+"
    r"high_n\s+(?P<high_n>[-+]?\d*\.?\d+),\s+"
    r"high_delta\s+(?P<high_delta>[-+]?\d*\.?\d+),\s+"
    r"med_n\s+(?P<med_n>[-+]?\d*\.?\d+),\s+"
    r"med_delta\s+(?P<med_delta>[-+]?\d*\.?\d+),\s+"
    r"low_n\s+(?P<low_n>[-+]?\d*\.?\d+),\s+"
    r"low_delta\s+(?P<low_delta>[-+]?\d*\.?\d+)"
)

EVAL_RE = re.compile(
    r"step\s+(?P<step>\d+):\s+train loss\s+(?P<train_loss>[-+]?\d*\.?\d+),\s+val loss\s+(?P<val_loss>[-+]?\d*\.?\d+)"
)


def parse_log(log_path: Path):
    rows = []
    eval_rows = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                d = {k: float(v) for k, v in m.groupdict().items()}
                d["iter"] = int(d["iter"])
                d["top1_gap"] = d["ic_top1"] - d["base_top1"]
                d["loss_gap"] = d["ic_loss"] - d["base_loss"]
                rows.append(d)

            e = EVAL_RE.search(line)
            if e:
                d = {k: float(v) for k, v in e.groupdict().items()}
                d["step"] = int(d["step"])
                eval_rows.append(d)

    if not rows:
        raise RuntimeError(
            f"No confidence-bucket IC rows found in {log_path}. "
            "Make sure the log contains high_n/high_delta/med_n/med_delta/low_n/low_delta."
        )

    return rows, eval_rows


def get(rows, key):
    return [r[key] for r in rows]


def savefig(out_dir: Path, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / filename
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"saved: {out}")


def plot_loss(rows, eval_rows, out_dir):
    it = get(rows, "iter")

    plt.figure(figsize=(8, 5))
    plt.plot(it, get(rows, "base_loss"), marker="o", label="base_loss")
    plt.plot(it, get(rows, "ic_loss"), marker="o", label="ic_loss")

    if eval_rows:
        steps = [r["step"] for r in eval_rows]
        plt.plot(steps, [r["train_loss"] for r in eval_rows], marker="s", linestyle="--", label="eval_train_loss")
        plt.plot(steps, [r["val_loss"] for r in eval_rows], marker="s", linestyle="--", label="eval_val_loss")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "01_loss_curves.png")


def plot_accuracy(rows, out_dir):
    it = get(rows, "iter")

    plt.figure(figsize=(8, 5))
    plt.plot(it, get(rows, "base_top1"), marker="o", label="base_top1")
    plt.plot(it, get(rows, "ic_top1"), marker="o", label="ic_top1")
    plt.xlabel("Iteration")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Base vs IC Top-1 Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "02_accuracy_curves.png")

    plt.figure(figsize=(8, 5))
    plt.axhline(0, linewidth=1)
    plt.plot(it, get(rows, "top1_gap"), marker="o", label="ic_top1 - base_top1")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy Gap")
    plt.title("IC Accuracy Improvement")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "03_accuracy_gap.png")


def plot_kv_stats(rows, out_dir):
    it = get(rows, "iter")

    plt.figure(figsize=(8, 5))
    plt.plot(it, get(rows, "delta_k_norm"), marker="o", label="delta_k_norm")
    plt.plot(it, get(rows, "delta_v_norm"), marker="o", label="delta_v_norm")
    plt.xlabel("Iteration")
    plt.ylabel("Norm")
    plt.title("IC Perturbation Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "04_delta_norm.png")

    plt.figure(figsize=(8, 5))
    plt.plot(it, get(rows, "gate_k_mean"), marker="o", label="gate_k_mean")
    plt.plot(it, get(rows, "gate_v_mean"), marker="o", label="gate_v_mean")
    plt.xlabel("Iteration")
    plt.ylabel("Gate Mean")
    plt.title("IC Gate Activation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "05_gate_mean.png")


def plot_confidence_delta(rows, out_dir):
    it = get(rows, "iter")

    plt.figure(figsize=(8, 5))
    plt.axhline(0, linewidth=1)
    plt.plot(it, get(rows, "high_delta"), marker="o", label="high_delta")
    plt.plot(it, get(rows, "med_delta"), marker="o", label="medium_delta")
    plt.plot(it, get(rows, "low_delta"), marker="o", label="low_delta")
    plt.xlabel("Iteration")
    plt.ylabel("IC Accuracy Delta")
    plt.title("IC Improvement by Baseline Confidence Bucket")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "06_confidence_bucket_delta.png")


def plot_confidence_counts(rows, out_dir):
    it = get(rows, "iter")

    plt.figure(figsize=(8, 5))
    plt.plot(it, get(rows, "high_n"), marker="o", label="high_n")
    plt.plot(it, get(rows, "med_n"), marker="o", label="medium_n")
    plt.plot(it, get(rows, "low_n"), marker="o", label="low_n")
    plt.xlabel("Iteration")
    plt.ylabel("Token Count")
    plt.title("Token Count by Confidence Bucket")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "07_confidence_bucket_counts.png")


def plot_weighted_contribution(rows, out_dir):
    it = get(rows, "iter")

    high_contrib = []
    med_contrib = []
    low_contrib = []
    total_gap = []

    for r in rows:
        total = max(r["high_n"] + r["med_n"] + r["low_n"], 1.0)
        high_contrib.append(r["high_delta"] * r["high_n"] / total)
        med_contrib.append(r["med_delta"] * r["med_n"] / total)
        low_contrib.append(r["low_delta"] * r["low_n"] / total)
        total_gap.append(r["top1_gap"])

    plt.figure(figsize=(8, 5))
    plt.axhline(0, linewidth=1)
    plt.plot(it, high_contrib, marker="o", label="high contribution")
    plt.plot(it, med_contrib, marker="o", label="medium contribution")
    plt.plot(it, low_contrib, marker="o", label="low contribution")
    plt.plot(it, total_gap, marker="s", linestyle="--", label="total ic-base gap")
    plt.xlabel("Iteration")
    plt.ylabel("Weighted Accuracy Contribution")
    plt.title("Which Confidence Bucket Explains the Total IC Gain?")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir, "08_confidence_weighted_contribution.png")


def plot_bucket_final_bar(rows, out_dir):
    last = rows[-1]

    labels = ["High", "Medium", "Low"]
    deltas = [last["high_delta"], last["med_delta"], last["low_delta"]]
    counts = [last["high_n"], last["med_n"], last["low_n"]]

    plt.figure(figsize=(8, 5))
    plt.axhline(0, linewidth=1)
    bars = plt.bar(labels, deltas)
    plt.xlabel("Confidence Bucket")
    plt.ylabel("IC Accuracy Delta")
    plt.title(f"Final Bucket Delta at Iter {last['iter']}")
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        y = height + 0.005 if height >= 0 else height - 0.02
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"n={int(count)}",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )
    plt.grid(True, axis="y", alpha=0.3)
    savefig(out_dir, "09_final_bucket_delta_bar.png")


def print_summary(rows, eval_rows, log_path, out_dir):
    last = rows[-1]
    print("\n=== Summary ===")
    print(f"log: {log_path}")
    print(f"output dir: {out_dir}")
    print(f"final iter: {last['iter']}")
    print(f"final base_loss: {last['base_loss']:.4f}")
    print(f"final ic_loss: {last['ic_loss']:.4f}")
    print(f"final base_top1: {last['base_top1']:.4f}")
    print(f"final ic_top1: {last['ic_top1']:.4f}")
    print(f"final top1 gap: {last['top1_gap']:.4f}")
    print(f"final high: n={last['high_n']:.0f}, delta={last['high_delta']:.4f}")
    print(f"final med:  n={last['med_n']:.0f}, delta={last['med_delta']:.4f}")
    print(f"final low:  n={last['low_n']:.0f}, delta={last['low_delta']:.4f}")

    if eval_rows:
        best_val = min(eval_rows, key=lambda r: r["val_loss"])
        final_eval = eval_rows[-1]
        print(f"best eval val loss: {best_val['val_loss']:.4f} at step {best_val['step']}")
        print(f"final eval train loss: {final_eval['train_loss']:.4f}")
        print(f"final eval val loss: {final_eval['val_loss']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to IC log file, e.g. logs/v3.9_v_only_conf_bucket.txt")
    parser.add_argument("--out", default=None, help="Optional output directory. Default: plots/<log stem>")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    out_dir = Path(args.out) if args.out else Path("plots") / log_path.stem

    rows, eval_rows = parse_log(log_path)

    plot_loss(rows, eval_rows, out_dir)
    plot_accuracy(rows, out_dir)
    plot_kv_stats(rows, out_dir)
    plot_confidence_delta(rows, out_dir)
    plot_confidence_counts(rows, out_dir)
    plot_weighted_contribution(rows, out_dir)
    plot_bucket_final_bar(rows, out_dir)
    print_summary(rows, eval_rows, log_path, out_dir)


if __name__ == "__main__":
    main()
