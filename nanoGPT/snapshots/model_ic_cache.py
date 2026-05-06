"""
Full definition of a GPT Language Model, all of it in this single file.

IC-cache version based on the original nanoGPT model.py.

Key idea:
- Do NOT modify final logits directly.
- Add a learnable IC memory path inside causal self-attention.
- For each source position s, build interference-cancellation keys/values
  K_ic[s], V_ic[s] from the original K,V and the selected / strongest-unselected token embeddings.
- Attention then attends over original memory plus IC memory:
      Attn(Q, [K, K_ic], [V, V_ic])
  with a causal mask such that position t can only see IC entries from s < t.
- This is a clean first implementation of "IC as memory editing / memory augmentation".

If use_ic_cache=False, behavior is the original nanoGPT behavior.
To enable IC-cache from command line, train.py must pass the new GPTConfig fields into model_args.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ICCacheEditor(nn.Module):
    """Learnable K_ic / V_ic generator."""

    def __init__(self, config):
        super().__init__()
        C = config.n_embd
        H = config.ic_cache_hidden_dim if config.ic_cache_hidden_dim > 0 else 2 * C

        self.k_mlp = nn.Sequential(
            nn.Linear(4 * C, H, bias=config.bias),
            nn.GELU(),
            nn.Linear(H, C, bias=config.bias),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(4 * C, H, bias=config.bias),
            nn.GELU(),
            nn.Linear(H, C, bias=config.bias),
        )
        self.gate_k = nn.Sequential(nn.Linear(4 * C, C, bias=config.bias), nn.Sigmoid())
        self.gate_v = nn.Sequential(nn.Linear(4 * C, C, bias=config.bias), nn.Sigmoid())
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, k_full, v_full, e_selected, e_unselected, alpha_k=0.05, alpha_v=0.10):
        z = torch.cat([k_full, v_full, e_selected, e_unselected], dim=-1)
        delta_k = self.k_mlp(z)
        delta_v = self.v_mlp(z)
        gate_k = self.gate_k(z)
        gate_v = self.gate_v(z)
        k_ic = k_full + alpha_k * self.dropout(gate_k * delta_k)
        v_ic = v_full + alpha_v * self.dropout(gate_v * delta_v)
        return k_ic, v_ic, delta_k, delta_v, gate_k, gate_v


class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_id=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_id = layer_id

        self.use_ic_cache = config.use_ic_cache
        self.ic_cache_mode = config.ic_cache_mode
        self.ic_cache_alpha_k = config.ic_cache_alpha_k
        self.ic_cache_alpha_v = config.ic_cache_alpha_v
        self.ic_editor = ICCacheEditor(config) if self.use_ic_cache else None

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash or self.use_ic_cache:
            if not self.flash:
                print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def _shape_to_heads(self, x, B, T, C):
        return x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    def forward(self, x, ic_e_selected=None, ic_e_unselected=None, return_ic_stats=False):
        B, T, C = x.size()

        q_full, k_full, v_full = self.c_attn(x).split(self.n_embd, dim=2)
        q = self._shape_to_heads(q_full, B, T, C)
        k = self._shape_to_heads(k_full, B, T, C)
        v = self._shape_to_heads(v_full, B, T, C)

        ic_stats = None

        if (not self.use_ic_cache) or ic_e_selected is None or ic_e_unselected is None:
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
        else:
            k_ic_full, v_ic_full, delta_k, delta_v, gate_k, gate_v = self.ic_editor(
                k_full, v_full, ic_e_selected, ic_e_unselected,
                alpha_k=self.ic_cache_alpha_k,
                alpha_v=self.ic_cache_alpha_v,
            )

            if self.ic_cache_mode == "residual":
                k = self._shape_to_heads(k_ic_full, B, T, C)
                v = self._shape_to_heads(v_ic_full, B, T, C)
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v
            elif self.ic_cache_mode == "append":
                k_ic = self._shape_to_heads(k_ic_full, B, T, C)
                v_ic = self._shape_to_heads(v_ic_full, B, T, C)
                k_all = torch.cat([k, k_ic], dim=2)
                v_all = torch.cat([v, v_ic], dim=2)
                att = (q @ k_all.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

                # original memory visible for s <= t; IC memory visible only for s < t
                mask_orig = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                mask_ic = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
                mask = torch.cat([mask_orig, mask_ic], dim=1).view(1, 1, T, 2 * T)
                att = att.masked_fill(~mask, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v_all
            else:
                raise ValueError(f"Unknown ic_cache_mode: {self.ic_cache_mode}. Use 'append' or 'residual'.")

            if return_ic_stats:
                with torch.no_grad():
                    ic_stats = {
                        "delta_k_norm": delta_k.norm(dim=-1).mean(),
                        "delta_v_norm": delta_v.norm(dim=-1).mean(),
                        "gate_k_mean": gate_k.mean(),
                        "gate_v_mean": gate_v.mean(),
                    }

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        if return_ic_stats:
            return y, ic_stats
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, layer_id=None):
        super().__init__()
        self.layer_id = layer_id
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, ic_e_selected=None, ic_e_unselected=None, return_ic_stats=False):
        if return_ic_stats:
            attn_out, ic_stats = self.attn(
                self.ln_1(x),
                ic_e_selected=ic_e_selected,
                ic_e_unselected=ic_e_unselected,
                return_ic_stats=True,
            )
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, ic_stats

        x = x + self.attn(
            self.ln_1(x),
            ic_e_selected=ic_e_selected,
            ic_e_unselected=ic_e_unselected,
            return_ic_stats=False,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # ---- IC-cache options ----
    use_ic_cache: bool = False
    ic_cache_mode: str = "append"  # append or residual
    ic_cache_hidden_dim: int = 0
    ic_cache_alpha_k: float = 0.05
    ic_cache_alpha_v: float = 0.10
    ic_cache_teacher_forcing: bool = True
    ic_cache_detach_base: bool = True
    ic_cache_lambda_base: float = 1.0
    ic_cache_lambda_ic: float = 1.0
    ic_cache_collect_stats: bool = True


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.last_ic_cache_stats = {}

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, layer_id=i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _masked_cross_entropy(logits, targets):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    @staticmethod
    def _get_strongest_unselected_ids(logits, selected_ids):
        with torch.no_grad():
            B, T, V = logits.shape
            k = 2 if V >= 2 else 1
            topk_ids = torch.topk(logits, k=k, dim=-1).indices
            best = topk_ids[..., 0]
            second = best if k == 1 else topk_ids[..., 1]
            strongest_unselected = torch.where(best == selected_ids, second, best)
        return strongest_unselected

    def _run_transformer(self, idx, ic_selected_ids=None, ic_unselected_ids=None, collect_ic_stats=False):
        device = idx.device
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        ic_e_selected = None
        ic_e_unselected = None
        if ic_selected_ids is not None and ic_unselected_ids is not None:
            ic_e_selected = self.transformer.wte(ic_selected_ids)
            ic_e_unselected = self.transformer.wte(ic_unselected_ids)

        collected = []
        for i, block in enumerate(self.transformer.h):
            want_stats = collect_ic_stats and self.config.ic_cache_collect_stats and (i == len(self.transformer.h) - 1)
            if want_stats:
                x, stats = block(x, ic_e_selected=ic_e_selected, ic_e_unselected=ic_e_unselected, return_ic_stats=True)
                if stats is not None:
                    collected.append(stats)
            else:
                x = block(x, ic_e_selected=ic_e_selected, ic_e_unselected=ic_e_unselected, return_ic_stats=False)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return x, logits, collected

    def _build_ic_token_context(self, idx, base_logits, targets=None):
        valid = torch.ones_like(idx, dtype=torch.bool)
        if targets is not None:
            valid = targets != -1

        if targets is not None and self.config.ic_cache_teacher_forcing:
            selected_ids = torch.where(valid, targets, torch.zeros_like(targets))
        else:
            with torch.no_grad():
                selected_ids = torch.argmax(base_logits, dim=-1)

        if self.config.ic_cache_detach_base:
            selected_ids = selected_ids.detach()
        unselected_ids = self._get_strongest_unselected_ids(base_logits.detach(), selected_ids)
        return selected_ids, unselected_ids

    def _summarize_ic_stats(self, stats_list, base_loss=None, ic_loss=None, base_logits=None, ic_logits=None, targets=None):
        if not stats_list:
            self.last_ic_cache_stats = {}
            return
        stats = stats_list[-1]
        out = {
            "delta_k_norm": float(stats["delta_k_norm"].detach().item()),
            "delta_v_norm": float(stats["delta_v_norm"].detach().item()),
            "gate_k_mean": float(stats["gate_k_mean"].detach().item()),
            "gate_v_mean": float(stats["gate_v_mean"].detach().item()),
        }
        if base_loss is not None:
            out["base_loss"] = float(base_loss.detach().item())
        if ic_loss is not None:
            out["ic_loss"] = float(ic_loss.detach().item())
        if base_logits is not None and ic_logits is not None and targets is not None:
            with torch.no_grad():
                valid = targets != -1
                safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
                denom = valid.float().sum().clamp_min(1.0)
                base_pred = torch.argmax(base_logits, dim=-1)
                ic_pred = torch.argmax(ic_logits, dim=-1)
                out["base_top1"] = float((((base_pred == safe_targets) & valid).float().sum() / denom).item())
                out["ic_top1"] = float((((ic_pred == safe_targets) & valid).float().sum() / denom).item())
        self.last_ic_cache_stats = out

    def forward(self, idx, targets=None):
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        if not self.config.use_ic_cache:
            _, logits, _ = self._run_transformer(idx)
            if targets is not None:
                loss = self._masked_cross_entropy(logits, targets)
            else:
                logits = logits[:, [-1], :]
                loss = None
            return logits, loss

        # IC-cache mode: base pass builds selection context; IC pass uses IC memory.
        with torch.set_grad_enabled(self.training and (not self.config.ic_cache_detach_base)):
            _, base_logits, _ = self._run_transformer(idx)

        base_logits_for_context = base_logits.detach() if self.config.ic_cache_detach_base else base_logits
        selected_ids, unselected_ids = self._build_ic_token_context(idx, base_logits_for_context, targets=targets)

        _, ic_logits, stats_list = self._run_transformer(
            idx,
            ic_selected_ids=selected_ids,
            ic_unselected_ids=unselected_ids,
            collect_ic_stats=True,
        )

        if targets is not None:
            base_loss = self._masked_cross_entropy(base_logits, targets)
            ic_loss = self._masked_cross_entropy(ic_logits, targets)
            loss = self.config.ic_cache_lambda_base * base_loss + self.config.ic_cache_lambda_ic * ic_loss
            self._summarize_ic_stats(stats_list, base_loss, ic_loss, base_logits, ic_logits, targets)
            return ic_logits, loss

        self._summarize_ic_stats(stats_list)
        logits = ic_logits[:, [-1], :]
        loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys_hf:
            if k not in sd:
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
