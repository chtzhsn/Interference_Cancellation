
"""
train.py (IC-cache version)

Supports:
- baseline GPT
- IC-cache (append / residual)
"""

import os
import time
import math
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default config (can override from command line)
# -----------------------------------------------------------------------------

out_dir = 'out'
eval_interval = 50
log_interval = 10
eval_iters = 20
max_iters = 300

batch_size = 64
block_size = 256

learning_rate = 1e-3
device = 'cpu'
compile = False

# ---- IC CONFIG ----
use_ic_cache = False
ic_cache_mode = "append"      # "append" or "residual"
ic_cache_alpha_k = 0.05
ic_cache_alpha_v = 0.10

# -----------------------------------------------------------------------------

# override via CLI
config_keys = [k for k in globals().keys() if not k.startswith('_')]
exec(open('configurator.py').read())

# -----------------------------------------------------------------------------

torch.manual_seed(1337)

# fake dataset (tiny shakespeare style)
data = torch.randint(0, 65, (10000,), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------

model_args = dict(
    block_size=block_size,
    vocab_size=65,
    n_layer=6,
    n_head=6,
    n_embd=384,

    # IC
    use_ic_cache=use_ic_cache,
    ic_cache_mode=ic_cache_mode,
    ic_cache_alpha_k=ic_cache_alpha_k,
    ic_cache_alpha_v=ic_cache_alpha_v,
)

model = GPT(GPTConfig(**model_args))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -----------------------------------------------------------------------------

def estimate_loss():
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y = get_batch()
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# -----------------------------------------------------------------------------

print("start training...")

for iter in range(max_iters):

    X, Y = get_batch()

    logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # logging
    if iter % log_interval == 0:
        print(f"iter {iter}: loss {loss.item():.4f}", end="")

        if use_ic_cache:
            stats = model.last_ic_cache_stats
            if stats:
                print(
                    f", base_loss {stats.get('base_loss',0):.4f}, "
                    f"ic_loss {stats.get('ic_loss',0):.4f}, "
                    f"delta_k {stats.get('delta_k_norm',0):.2f}, "
                    f"gate {stats.get('gate_k_mean',0):.3f}, "
                    f"base_top1 {stats.get('base_top1',0):.3f}, "
                    f"ic_top1 {stats.get('ic_top1',0):.3f}",
                    end=""
                )
        print()

    # evaluation
    if iter % eval_interval == 0:
        val_loss = estimate_loss()
        print(f"eval loss: {val_loss:.4f}")

# -----------------------------------------------------------------------------

print("training finished")
