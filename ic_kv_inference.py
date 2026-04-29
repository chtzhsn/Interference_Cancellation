"""
ic_kv_inference.py

Standalone step-by-step IC-KV inference runner for the IC-cache nanoGPT model.

Place this file at:
    /mnt/d/MR_projects/InterferenceCancellation/ic_kv_inference.py

Run from the project root:
    python ic_kv_inference.py

This script assumes nanoGPT/model.py is the IC-cache version that defines:
    GPTConfig(use_ic_cache=True, ...)
    CausalSelfAttention.ic_editor
"""

import math
import torch
import torch.nn.functional as F

from nanoGPT.model import GPT, GPTConfig


class ICKVRunner:
    """
    Step-by-step autoregressive runner with persistent KV cache and IC updates.
    """

    def __init__(self, model: GPT, device: str = "cpu"):
        self.model = model
        self.config = model.config
        self.device = device
        self.reset_cache()

    def reset_cache(self):
        self.k_cache = []
        self.v_cache = []
        self.last_logits = None

    def _select_tokens_from_last_logits(self):
        assert self.last_logits is not None
        logits = self.last_logits  # (B, vocab)
        sel = torch.argmax(logits, dim=-1)  # (B,)
        top2 = torch.topk(logits, k=2, dim=-1).indices
        best = top2[:, 0]
        second = top2[:, 1]
        unsel = torch.where(best == sel, second, best)
        return sel, unsel

    def _edit_kv_full(self, block, k_full, v_full):
        """
        Edit full-dimensional k/v before reshaping into attention heads.

        k_full, v_full: (B, 1, C)
        """
        if self.last_logits is None:
            return k_full, v_full

        if not hasattr(block.attn, "ic_editor") or block.attn.ic_editor is None:
            raise AttributeError(
                "block.attn.ic_editor does not exist. "
                "Make sure model was created with GPTConfig(use_ic_cache=True)."
            )

        sel, unsel = self._select_tokens_from_last_logits()
        e_sel = self.model.transformer.wte(sel).unsqueeze(1)      # (B,1,C)
        e_unsel = self.model.transformer.wte(unsel).unsqueeze(1)  # (B,1,C)

        k_ic_full, v_ic_full, delta_k, delta_v, gate_k, gate_v = block.attn.ic_editor(
            k_full,
            v_full,
            e_sel,
            e_unsel,
            alpha_k=getattr(self.config, "ic_cache_alpha_k", 0.05),
            alpha_v=getattr(self.config, "ic_cache_alpha_v", 0.10),
        )
        return k_ic_full, v_ic_full

    def step(self, idx):
        """
        Run one autoregressive step.

        idx: LongTensor, shape (B, T), full generated sequence so far.
             Only the last token is consumed by this step runner.

        returns:
            logits: FloatTensor, shape (B, vocab_size)
        """
        B, T = idx.shape
        device = idx.device
        C = self.config.n_embd

        # Input embedding for newest token only
        tok_emb = self.model.transformer.wte(idx[:, -1])  # (B,C)
        pos = torch.tensor([T - 1], dtype=torch.long, device=device)
        pos_emb = self.model.transformer.wpe(pos)         # (1,C)
        x = tok_emb.unsqueeze(1) + pos_emb.unsqueeze(0)   # (B,1,C)
        x = self.model.transformer.drop(x)

        for layer_id, block in enumerate(self.model.transformer.h):
            x_ln = block.ln_1(x)

            # q/k/v in full dimension first: (B,1,C)
            q_full, k_full, v_full = block.attn.c_attn(x_ln).split(C, dim=2)

            # IC edit in full C dimension before head reshape
            k_full, v_full = self._edit_kv_full(block, k_full, v_full)

            nh = block.attn.n_head
            hs = C // nh

            q = q_full.view(B, 1, nh, hs).transpose(1, 2)  # (B,nh,1,hs)
            k = k_full.view(B, 1, nh, hs).transpose(1, 2)  # (B,nh,1,hs)
            v = v_full.view(B, 1, nh, hs).transpose(1, 2)  # (B,nh,1,hs)

            # Append edited k/v to persistent cache
            if len(self.k_cache) <= layer_id:
                self.k_cache.append(k)
                self.v_cache.append(v)
            else:
                self.k_cache[layer_id] = torch.cat([self.k_cache[layer_id], k], dim=2)
                self.v_cache[layer_id] = torch.cat([self.v_cache[layer_id], v], dim=2)

            K = self.k_cache[layer_id]
            V = self.v_cache[layer_id]

            # One-token causal attention over cache
            att = (q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = F.softmax(att, dim=-1)
            y = att @ V

            y = y.transpose(1, 2).contiguous().view(B, 1, C)
            y = block.attn.resid_dropout(block.attn.c_proj(y))

            x = x + y
            x = x + block.mlp(block.ln_2(x))

        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x[:, -1, :])

        # Store logits for next step's selected/unselected tokens.
        self.last_logits = logits.detach()
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None, greedy=True):
        """
        Generate tokens step-by-step using persistent IC-KV cache.
        """
        self.reset_cache()

        # Prefill cache token by token for the initial prompt.
        for pos in range(idx.size(1)):
            _ = self.step(idx[:, :pos + 1])

        out = idx.clone()

        for _ in range(max_new_tokens):
            logits = self.step(out)
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], -float("Inf"))

            if greedy:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_token], dim=1)

        return out


def decode_token_ids(token_ids):
    """Fallback display helper."""
    return " ".join(str(int(x)) for x in token_ids)


def main():
    device = "cpu"

    model_args = dict(
        block_size=256,
        vocab_size=65,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=False,

        # Required for IC editor modules inside attention.
        use_ic_cache=True,
        ic_cache_mode="residual",
        ic_cache_alpha_k=0.05,
        ic_cache_alpha_v=0.10,
    )

    model = GPT(GPTConfig(**model_args))
    model.to(device)
    model.eval()

    runner = ICKVRunner(model, device=device)

    # Start from one random token.
    idx = torch.randint(0, 65, (1, 1), dtype=torch.long, device=device)

    print("Start token:", idx.item())
    print("Generating with persistent IC-KV cache...")

    out = runner.generate(
        idx,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        greedy=True,
    )

    print("Generated token ids:")
    print(out[0].tolist())
    print("As ids:")
    print(decode_token_ids(out[0]))


if __name__ == "__main__":
    main()
