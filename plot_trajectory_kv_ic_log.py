"""
plot_trajectory_kv_ic_log.py

Parse trajectory-wide KV IC training log and generate report figures.

Expected input:
    logs/trajectory_kv_ic_100.txt

Run from project root:
    python plot_trajectory_kv_ic_log.py

Outputs:
    plots/v3_trajectory_kv_ic/
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt


# LOG_PATH = Path("logs/v3.3_ic_only.txt")
# OUT_DIR = Path("plots/v3.3_ic_only")

# LOG_PATH = Path("logs/v3.4_traj_condi_ic.txt")
# OUT_DIR = Path("plots/v3.4_traj_condi_ic")

LOG_PATH = Path("logs/v3.5_traj_margin_ic.txt")
OUT_DIR = Path("plots/v3.5_traj_margin_ic")

ITER_RE = re.compile(
    r"iter\s+(?P<iter>\d+):\s+loss\s+(?P<loss>[-+]?\d*\.?\d+),\s+"
    r"base_loss\s+(?P<base_loss>[-+]?\d*\.?\d+),\s+"
    r"ic_loss\s+(?P<ic_loss>[-+]?\d*\.?\d+),\s+"
    r"delta_k_norm\s+(?P<delta_k_norm>[-+]?\d*\.?\d+),\s+"
    r"delta_v_norm\s+(?P<delta_v_norm>[-+]?\d*\.?\d+),\s+"
    r"gate_k_mean\s+(?P<gate_k_mean>[-+]?\d*\.?\d+),\s+"
    r"gate_v_mean\s+(?P<gate_v_mean>[-+]?\d*\.?\d+),\s+"
    r"base_top1\s+(?P<base_top1>[-+]?\d*\.?\d+),\s+"
    r"ic_top1\s+(?P<ic_top1>[-+]?\d*\.?\d+)"
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
                d["loss_gap_ic_minus_base"] = d["ic_loss"] - d["base_loss"]
                d["top1_gap_ic_minus_base"] = d["ic_top1"] - d["base_top1"]
                rows.append(d)

            e = EVAL_RE.search(line)
            if e:
                d = {k: float(v) for k, v in e.groupdict().items()}
                d["step"] = int(d["step"])
                eval_rows.append(d)

    if not rows:
        raise RuntimeError(f"No IC training rows found in {log_path}")

    return rows, eval_rows


def get(rows, key):
    return [r[key] for r in rows]


def savefig(name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"saved: {out}")


def plot_loss(rows, eval_rows):
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
    plt.title("Trajectory-wide KV IC: Base vs IC Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("trajectory_kv_ic_loss.png")


def plot_accuracy(rows):
    it = get(rows, "iter")
    plt.figure(figsize=(8, 5))
    plt.plot(it, get(rows, "base_top1"), marker="o", label="base_top1")
    plt.plot(it, get(rows, "ic_top1"), marker="o", label="ic_top1")
    plt.xlabel("Iteration")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Trajectory-wide KV IC: Base vs IC Top-1 Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("trajectory_kv_ic_accuracy.png")


def plot_kv_stats(rows):
    it = get(rows, "iter")
    plt.figure(figsize=(8, 5))
    plt.plot(it, get(rows, "delta_k_norm"), marker="o", label="delta_k_norm")
    plt.plot(it, get(rows, "delta_v_norm"), marker="o", label="delta_v_norm")
    plt.xlabel("Iteration")
    plt.ylabel("Norm")
    plt.title("Trajectory-wide KV IC: KV Correction Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("trajectory_kv_ic_delta_norm.png")

    plt.figure(figsize=(8, 5))
    plt.plot(it, get(rows, "gate_k_mean"), marker="o", label="gate_k_mean")
    plt.plot(it, get(rows, "gate_v_mean"), marker="o", label="gate_v_mean")
    plt.xlabel("Iteration")
    plt.ylabel("Gate Mean")
    plt.title("Trajectory-wide KV IC: Gate Activation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("trajectory_kv_ic_gate_mean.png")


def plot_gap(rows):
    it = get(rows, "iter")
    plt.figure(figsize=(8, 5))
    plt.axhline(0, linewidth=1)
    plt.plot(it, get(rows, "loss_gap_ic_minus_base"), marker="o", label="ic_loss - base_loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss Gap")
    plt.title("Trajectory-wide KV IC: IC Loss Gap")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("trajectory_kv_ic_loss_gap.png")

    plt.figure(figsize=(8, 5))
    plt.axhline(0, linewidth=1)
    plt.plot(it, get(rows, "top1_gap_ic_minus_base"), marker="o", label="ic_top1 - base_top1")
    plt.xlabel("Iteration")
    plt.ylabel("Top-1 Accuracy Gap")
    plt.title("Trajectory-wide KV IC: IC Accuracy Gap")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("trajectory_kv_ic_accuracy_gap.png")


def print_summary(rows, eval_rows):
    last = rows[-1]
    print("\n=== Summary ===")
    print(f"Final iter: {last['iter']}")
    print(f"Final base_loss: {last['base_loss']:.4f}")
    print(f"Final ic_loss: {last['ic_loss']:.4f}")
    print(f"Final ic_loss - base_loss: {last['loss_gap_ic_minus_base']:.4f}")
    print(f"Final delta_k_norm: {last['delta_k_norm']:.4f}")
    print(f"Final delta_v_norm: {last['delta_v_norm']:.4f}")
    print(f"Final gate_k_mean: {last['gate_k_mean']:.4f}")
    print(f"Final gate_v_mean: {last['gate_v_mean']:.4f}")
    print(f"Final base_top1: {last['base_top1']:.4f}")
    print(f"Final ic_top1: {last['ic_top1']:.4f}")
    print(f"Final ic_top1 - base_top1: {last['top1_gap_ic_minus_base']:.4f}")
    if eval_rows:
        best_val = min(eval_rows, key=lambda r: r["val_loss"])
        final_eval = eval_rows[-1]
        print(f"Best eval val loss: {best_val['val_loss']:.4f} at step {best_val['step']}")
        print(f"Final eval train loss: {final_eval['train_loss']:.4f}")
        print(f"Final eval val loss: {final_eval['val_loss']:.4f}")


def main():
    rows, eval_rows = parse_log(LOG_PATH)
    plot_loss(rows, eval_rows)
    plot_accuracy(rows)
    plot_kv_stats(rows)
    plot_gap(rows)
    print_summary(rows, eval_rows)


if __name__ == "__main__":
    main()
