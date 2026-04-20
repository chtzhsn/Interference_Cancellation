import os
import re
import argparse
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def maybe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    x = x.strip()
    if x.lower() == "none":
        return None
    return float(x)


def parse_log_file(log_path: str) -> Dict[str, List[float]]:
    """
    Parse nanoGPT training logs with IC/probe metrics.

    Expected patterns include lines like:
      step 100: train loss 3.7788, val loss 3.7652
      iter 100: loss 3.7844, base_loss 2.6290, mit_loss 2.7936, margin_loss 0.6348,
                delta_norm 101.9627, gate_mean 0.3077,
                base_margin -0.7254, mit_margin -0.2896, margin_gain 0.4358,
                base_top1 0.2676, mit_top1 0.2285, ...
    """
    # Eval lines
    eval_re = re.compile(
        r"step\s+(\d+):\s+train loss\s+([-\d.]+),\s+val loss\s+([-\d.]+)"
    )

    # Iter lines: make each field optional except iter/loss
    iter_re = re.compile(
        r"iter\s+(\d+):\s+loss\s+([-\d.]+)"
        r"(?:,\s+base_loss\s+([-\d.]+))?"
        r"(?:,\s+mit_loss\s+([-\d.]+))?"
        r"(?:,\s+margin_loss\s+([-\d.]+))?"
        r"(?:,\s+delta_norm\s+([-\d.]+))?"
        r"(?:,\s+gate_mean\s+([-\d.]+))?"
        r"(?:,\s+base_margin\s+([-\d.]+))?"
        r"(?:,\s+mit_margin\s+([-\d.]+))?"
        r"(?:,\s+margin_gain\s+([-\d.]+))?"
        r"(?:,\s+base_top1\s+([-\d.]+))?"
        r"(?:,\s+mit_top1\s+([-\d.]+))?"
    )

    data = {
        "eval_step": [],
        "eval_train_loss": [],
        "eval_val_loss": [],
        "iter": [],
        "loss": [],
        "base_loss": [],
        "mit_loss": [],
        "margin_loss": [],
        "delta_norm": [],
        "gate_mean": [],
        "base_margin": [],
        "mit_margin": [],
        "margin_gain": [],
        "base_top1": [],
        "mit_top1": [],
    }

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m_eval = eval_re.search(line)
            if m_eval:
                data["eval_step"].append(int(m_eval.group(1)))
                data["eval_train_loss"].append(float(m_eval.group(2)))
                data["eval_val_loss"].append(float(m_eval.group(3)))
                continue

            m_iter = iter_re.search(line)
            if m_iter:
                groups = m_iter.groups()
                data["iter"].append(int(groups[0]))
                data["loss"].append(float(groups[1]))
                data["base_loss"].append(maybe_float(groups[2]))
                data["mit_loss"].append(maybe_float(groups[3]))
                data["margin_loss"].append(maybe_float(groups[4]))
                data["delta_norm"].append(maybe_float(groups[5]))
                data["gate_mean"].append(maybe_float(groups[6]))
                data["base_margin"].append(maybe_float(groups[7]))
                data["mit_margin"].append(maybe_float(groups[8]))
                data["margin_gain"].append(maybe_float(groups[9]))
                data["base_top1"].append(maybe_float(groups[10]))
                data["mit_top1"].append(maybe_float(groups[11]))
                continue

    return data


def filter_xy(xs: List[float], ys: List[Optional[float]]):
    x_out, y_out = [], []
    for x, y in zip(xs, ys):
        if y is not None:
            x_out.append(x)
            y_out.append(y)
    return x_out, y_out


def save_plot(fig, outdir: str, filename: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_curves(data: Dict[str, List[float]], outdir: str):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    if data["iter"]:
        ax.plot(data["iter"], data["loss"], label="total loss")

        x, y = filter_xy(data["iter"], data["base_loss"])
        if x:
            ax.plot(x, y, label="base_loss")

        x, y = filter_xy(data["iter"], data["mit_loss"])
        if x:
            ax.plot(x, y, label="mit_loss")

        x, y = filter_xy(data["iter"], data["margin_loss"])
        if x:
            ax.plot(x, y, label="margin_loss")

    if data["eval_step"]:
        ax.plot(data["eval_step"], data["eval_train_loss"], marker='o', linestyle='--', label="eval train loss")
        ax.plot(data["eval_step"], data["eval_val_loss"], marker='o', linestyle='--', label="eval val loss")

    ax.set_title("Loss Curves")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_plot(fig, outdir, "loss_curves.png")


def plot_margin_curves(data: Dict[str, List[float]], outdir: str):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    x, y = filter_xy(data["iter"], data["base_margin"])
    if x:
        ax.plot(x, y, label="base_margin")

    x, y = filter_xy(data["iter"], data["mit_margin"])
    if x:
        ax.plot(x, y, label="mit_margin")

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title("Target-Competitor Margin")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Margin")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_plot(fig, outdir, "margin_curves.png")


def plot_margin_gain(data: Dict[str, List[float]], outdir: str):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    x, y = filter_xy(data["iter"], data["margin_gain"])
    if x:
        ax.plot(x, y, label="margin_gain")

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title("Margin Gain")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("mit_margin - base_margin")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_plot(fig, outdir, "margin_gain.png")


def plot_top1_accuracy(data: Dict[str, List[float]], outdir: str):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    x, y = filter_xy(data["iter"], data["base_top1"])
    if x:
        ax.plot(x, y, label="base_top1")

    x, y = filter_xy(data["iter"], data["mit_top1"])
    if x:
        ax.plot(x, y, label="mit_top1")

    ax.set_title("Top-1 Accuracy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_plot(fig, outdir, "top1_accuracy.png")


def plot_ic_stats(data: Dict[str, List[float]], outdir: str):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    x, y = filter_xy(data["iter"], data["delta_norm"])
    if x:
        ax.plot(x, y, label="delta_norm")

    x, y = filter_xy(data["iter"], data["gate_mean"])
    if x:
        ax.plot(x, y, label="gate_mean")

    ax.set_title("IC Internal Statistics")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_plot(fig, outdir, "ic_stats.png")


def print_summary(data: Dict[str, List[float]]):
    print("\n=== Summary ===")
    if data["eval_step"]:
        best_idx = min(range(len(data["eval_val_loss"])), key=lambda i: data["eval_val_loss"][i])
        print(f"Best eval val loss: {data['eval_val_loss'][best_idx]:.4f} at step {data['eval_step'][best_idx]}")
        print(f"Final eval val loss: {data['eval_val_loss'][-1]:.4f} at step {data['eval_step'][-1]}")

    if data["iter"]:
        print(f"Final iter: {data['iter'][-1]}")
        print(f"Final total loss: {data['loss'][-1]:.4f}")

    for key in ["base_loss", "mit_loss", "margin_loss", "delta_norm", "gate_mean",
                "base_margin", "mit_margin", "margin_gain", "base_top1", "mit_top1"]:
        vals = [v for v in data[key] if v is not None]
        if vals:
            print(f"Final {key}: {vals[-1]:.4f}")


def default_outdir_from_log(log_path: str) -> str:
    stem = os.path.splitext(os.path.basename(log_path))[0]
    return os.path.join("plots", stem)


def main():
    parser = argparse.ArgumentParser(description="Plot nanoGPT training log metrics.")
    parser.add_argument("log_path", type=str, help="Path to training log txt file")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()

    outdir = args.outdir if args.outdir is not None else default_outdir_from_log(args.log_path)

    data = parse_log_file(args.log_path)

    if not data["iter"] and not data["eval_step"]:
        raise ValueError("No recognizable training log lines found. Check the log format/path.")

    plot_loss_curves(data, outdir)
    plot_margin_curves(data, outdir)
    plot_margin_gain(data, outdir)
    plot_top1_accuracy(data, outdir)
    plot_ic_stats(data, outdir)

    print_summary(data)


if __name__ == "__main__":
    main()