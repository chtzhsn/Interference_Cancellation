
"""
Full definition of a GPT Language Model, all of it in this single file.

This version adds a first-step, K+V KV-based interference-cancellation (KVIC)
surrogate for teacher-forcing training:

- only the last transformer block is KVIC-enabled
- both K and V are corrected
- selected token defaults to the teacher-forced target
- strongest unselected token is taken from a provisional logits head
  computed before the last block
- inference path remains baseline for now (no cache-time VIC yet)

This is intended as the safest first implementation before moving to true cache-edit / step-by-step inference-time KV rewriting.
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


class KeyValueInterferenceCorrection(nn.Module):
    """
    K+V interference correction module.

    Input:
        k_token     : (B, T, C), key vectors reassembled across heads
        v_token     : (B, T, C), value vectors reassembled across heads
        e_selected  : (B, T, C)
        e_unselected: (B, T, C)

    Output:
        corrected k_token / v_token, along with delta/gate for logging
    """

    def __init__(self, config):
        super().__init__()
        c = config.n_embd
        hidden = config.kvic_hidden_dim if config.kvic_hidden_dim > 0 else 2 * c

        self.delta_k_mlp = nn.Sequential(
            nn.Linear(3 * c, hidden, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden, c, bias=config.bias),
        )
        self.delta_v_mlp = nn.Sequential(
            nn.Linear(3 * c, hidden, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden, c, bias=config.bias),
        )

        self.gate_k = nn.Sequential(
            nn.Linear(3 * c, c, bias=config.bias),
            nn.Sigmoid(),
        )
        self.gate_v = nn.Sequential(
            nn.Linear(3 * c, c, bias=config.bias),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, k_token, v_token, e_selected, e_unselected, alpha_k, alpha_v):
        zk = torch.cat([k_token, e_selected, e_unselected], dim=-1)
        zv = torch.cat([v_token, e_selected, e_unselected], dim=-1)

        delta_k = self.delta_k_mlp(zk)
        delta_v = self.delta_v_mlp(zv)
        gate_k = self.gate_k(zk)
        gate_v = self.gate_v(zv)

        k_corr = k_token + alpha_k * self.dropout(gate_k * delta_k)
        v_corr = v_token + alpha_v * self.dropout(gate_v * delta_v)
        return k_corr, v_corr, delta_k, delta_v, gate_k, gate_v


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.use_kvic = config.use_kvic
        self.kvic_alpha_k = config.kvic_alpha_k
        self.kvic_alpha_v = config.kvic_alpha_v
        self.kvic_module = KeyValueInterferenceCorrection(config) if config.use_kvic else None

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x, kvic_info=None, return_kvic_stats=False):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        kvic_stats = None

        if self.use_kvic and kvic_info is not None:
            # reassemble key/value tokens to (B, T, C), correct them, then split back
            k_token = k.transpose(1, 2).contiguous().view(B, T, C)
            v_token = v.transpose(1, 2).contiguous().view(B, T, C)
            e_selected = kvic_info["e_selected"]
            e_unselected = kvic_info["e_unselected"]

            k_corr_token, v_corr_token, delta_k, delta_v, gate_k, gate_v = self.kvic_module(
                k_token, v_token, e_selected, e_unselected,
                alpha_k=self.kvic_alpha_k, alpha_v=self.kvic_alpha_v
            )
            k = k_corr_token.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v_corr_token.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            if return_kvic_stats:
                kvic_stats = {
                    "delta_k_norm": delta_k.norm(dim=-1).mean(),
                    "delta_v_norm": delta_v.norm(dim=-1).mean(),
                    "gate_k_mean": gate_k.mean(),
                    "gate_v_mean": gate_v.mean(),
                }

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

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if return_kvic_stats:
            return y, kvic_stats
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

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kvic_info=None, return_kvic_stats=False):
        if return_kvic_stats:
            attn_out, kvic_stats = self.attn(self.ln_1(x), kvic_info=kvic_info, return_kvic_stats=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, kvic_stats
        else:
            x = x + self.attn(self.ln_1(x), kvic_info=kvic_info)
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

    # ---- K+V KV-based IC options ----
    use_kvic: bool = False
    kvic_hidden_dim: int = 0
    kvic_alpha_k: float = 0.05
    kvic_alpha_v: float = 0.1
    kvic_lambda_base: float = 1.0
    kvic_lambda_ic: float = 0.3
    kvic_lambda_margin: float = 0.5
    kvic_margin_target: float = 0.2


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.last_kvic_stats = {}
        self.last_probe_stats = {}

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
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
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )

    @staticmethod
    def _get_strongest_unselected_ids(logits, selected_ids):
        """
        logits       : (B, T, V)
        selected_ids : (B, T)
        """
        with torch.no_grad():
            B, T, V = logits.shape
            k = 2 if V >= 2 else 1
            topk_ids = torch.topk(logits, k=k, dim=-1).indices
            best = topk_ids[..., 0]
            second = best if k == 1 else topk_ids[..., 1]
            return torch.where(best == selected_ids, second, best)

    @staticmethod
    def _gather_logits(logits, token_ids):
        return logits.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

    def _compute_probe_stats(self, base_logits, ic_logits, targets):
        with torch.no_grad():
            valid = targets != -1
            safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
            competitor_ids = self._get_strongest_unselected_ids(base_logits, safe_targets)

            base_target = self._gather_logits(base_logits, safe_targets)
            base_comp = self._gather_logits(base_logits, competitor_ids)
            vic_target = self._gather_logits(ic_logits, safe_targets)
            vic_comp = self._gather_logits(ic_logits, competitor_ids)

            base_margin = base_target - base_comp
            vic_margin = vic_target - vic_comp
            margin_gain = vic_margin - base_margin

            base_pred = torch.argmax(base_logits, dim=-1)
            vic_pred = torch.argmax(ic_logits, dim=-1)

            valid_float = valid.float()
            denom = valid_float.sum().clamp_min(1.0)

            base_top1 = ((base_pred == safe_targets) & valid).float().sum() / denom
            vic_top1 = ((vic_pred == safe_targets) & valid).float().sum() / denom

            base_margin_mean = (base_margin * valid_float).sum() / denom
            vic_margin_mean = (vic_margin * valid_float).sum() / denom
            margin_gain_mean = (margin_gain * valid_float).sum() / denom

        return {
            "base_margin": float(base_margin_mean.item()),
            "ic_margin": float(vic_margin_mean.item()),
            "margin_gain": float(margin_gain_mean.item()),
            "base_top1_acc": float(base_top1.item()),
            "ic_top1_acc": float(vic_top1.item()),
        }

    def _compute_margin_loss(self, logits, targets):
        valid = targets != -1
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
        competitor_ids = self._get_strongest_unselected_ids(logits, safe_targets)

        target_logits = self._gather_logits(logits, safe_targets)
        competitor_logits = self._gather_logits(logits, competitor_ids)

        margins = target_logits - competitor_logits
        losses = F.relu(self.config.kvic_margin_target - margins)

        valid_float = valid.float()
        denom = valid_float.sum().clamp_min(1.0)
        return (losses * valid_float).sum() / denom

    def _run_prefix_blocks(self, x):
        # all but the final block
        for block in self.transformer.h[:-1]:
            x = block(x)
        return x

    def _run_last_block_base(self, x):
        return self.transformer.h[-1](x)

    def _run_last_block_kvic(self, x, e_selected, e_unselected):
        kvic_info = {
            "e_selected": e_selected,
            "e_unselected": e_unselected,
        }
        x_ic, kvic_stats = self.transformer.h[-1](x, kvic_info=kvic_info, return_kvic_stats=True)
        return x_ic, kvic_stats

    def forward(self, idx, targets=None, return_aux=False):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # inference path: keep baseline behavior for now
        if targets is None or not self.config.use_kvic:
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)

            if targets is not None:
                logits = self.lm_head(x)
                loss = self._masked_cross_entropy(logits, targets)
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None

            if targets is not None and not self.config.use_kvic:
                self.last_kvic_stats = {
                    "base_loss": float(loss.detach().item()),
                    "ic_loss": None,
                    "margin_loss": None,
                    "delta_k_norm": None,
                    "delta_v_norm": None,
                    "gate_k_mean": None,
                    "gate_v_mean": None,
                }
                self.last_probe_stats = {}

            return logits, loss

        # ---- training/eval K+V IC surrogate path ----
        # 1) run prefix blocks (all except final)
        x_prefix = self._run_prefix_blocks(x)

        # 2) provisional logits before last block, used only to find strongest unselected
        provisional_hidden = self.transformer.ln_f(x_prefix)
        provisional_logits = self.lm_head(provisional_hidden)

        valid = targets != -1
        selected_ids = torch.where(valid, targets, torch.zeros_like(targets))
        unselected_ids = self._get_strongest_unselected_ids(provisional_logits, selected_ids)

        e_selected = self.transformer.wte(selected_ids)
        e_unselected = self.transformer.wte(unselected_ids)

        # 3) base path through last block
        x_base = self._run_last_block_base(x_prefix)
        x_base = self.transformer.ln_f(x_base)
        base_logits = self.lm_head(x_base)
        base_loss = self._masked_cross_entropy(base_logits, targets)

        # 4) K+V corrected path through last block
        x_ic, kvic_internal = self._run_last_block_kvic(x_prefix, e_selected, e_unselected)
        x_ic = self.transformer.ln_f(x_ic)
        ic_logits = self.lm_head(x_ic)
        ic_loss = self._masked_cross_entropy(ic_logits, targets)
        margin_loss = self._compute_margin_loss(ic_logits, targets)

        total_loss = (
            self.config.kvic_lambda_base * base_loss
            + self.config.kvic_lambda_ic * ic_loss
            + self.config.kvic_lambda_margin * margin_loss
        )

        probe_stats = self._compute_probe_stats(base_logits, ic_logits, targets)

        self.last_kvic_stats = {
            "base_loss": float(base_loss.detach().item()),
            "ic_loss": float(ic_loss.detach().item()),
            "margin_loss": float(margin_loss.detach().item()),
            "delta_k_norm": float(kvic_internal["delta_k_norm"].detach().item()),
            "delta_v_norm": float(kvic_internal["delta_v_norm"].detach().item()),
            "gate_k_mean": float(kvic_internal["gate_k_mean"].detach().item()),
            "gate_v_mean": float(kvic_internal["gate_v_mean"].detach().item()),
        }
        self.last_probe_stats = probe_stats

        if return_aux:
            aux = {
                "base_logits": base_logits,
                "ic_logits": ic_logits,
                "selected_ids": selected_ids,
                "unselected_ids": unselected_ids,
                "vic_stats": vic_internal,
                "probe_stats": probe_stats,
            }
            return ic_logits, total_loss, aux

        return ic_logits, total_loss

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

        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight',
        ]

        # Only copy overlapping parameters; KVIC-specific weights stay randomly initialized
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
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

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
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # For now, generation stays baseline; true inference-time KV rewriting comes next.
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)  # targets=None -> baseline path
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
