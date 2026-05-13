"""
Full definition of a GPT Language Model, all of it in this single file.

Trajectory-level IC (minimal viable version):
- standard base pass produces h_t and base logits
- selected token = teacher-forced target during training
- strongest unselected token = strongest non-target from base logits
- a TrajectoryInterferenceEditor produces a mitigated latent u_ic_t from h_t
- this mitigated latent is shifted by one step and injected into the next-step input embedding
- a second pass produces IC logits and losses

This is a first surrogate for trajectory-level hindsight correction before moving to
true inference-time cache-edit / rollout-based editing.
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
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TrajectoryInterferenceEditor(nn.Module):
    """
    Produces a mitigated latent token u_ic_t from current hidden state h_t,
    selected token embedding, and strongest unselected token embedding.

    u_ic_t is then shifted by one position and injected into the next-step input.
    """

    def __init__(self, config):
        super().__init__()
        c = config.n_embd
        hidden = config.traj_ic_hidden_dim if config.traj_ic_hidden_dim > 0 else 2 * c

        self.delta_mlp = nn.Sequential(
            nn.Linear(3 * c, hidden, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden, c, bias=config.bias),
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * c, c, bias=config.bias),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, h, e_selected, e_unselected, alpha):
        z = torch.cat([h, e_selected, e_unselected], dim=-1)
        delta_u = self.delta_mlp(z)
        gate_u = self.gate(z)
        u_ic = h + alpha * self.dropout(gate_u * delta_u)
        return u_ic, delta_u, gate_u


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # ---- trajectory-level IC options ----
    use_traj_ic: bool = False
    traj_ic_hidden_dim: int = 0
    traj_ic_alpha: float = 0.1
    traj_ic_lambda_base: float = 1.0
    traj_ic_lambda_ic: float = 0.3
    traj_ic_lambda_traj: float = 0.0
    traj_ic_lambda_margin: float = 0.5
    traj_ic_margin_target: float = 0.2
    # distribution-level IC
    traj_ic_lambda_dist: float = 0.0
    traj_ic_teacher_mode: str = "reference"


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.last_traj_ic_stats = {}
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

        self.traj_editor = TrajectoryInterferenceEditor(config) if config.use_traj_ic else None

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
            ignore_index=-1
        )

    @staticmethod
    def _gather_logits(logits, token_ids):
        return logits.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

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

    def _compute_probe_stats(self, base_logits, ic_logits, targets):
        with torch.no_grad():
            valid = targets != -1
            safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
            competitor_ids = self._get_strongest_unselected_ids(base_logits, safe_targets)

            base_target_logits = self._gather_logits(base_logits, safe_targets)
            base_comp_logits = self._gather_logits(base_logits, competitor_ids)

            ic_target_logits = self._gather_logits(ic_logits, safe_targets)
            ic_comp_logits = self._gather_logits(ic_logits, competitor_ids)

            base_margin = base_target_logits - base_comp_logits
            ic_margin = ic_target_logits - ic_comp_logits
            margin_gain = ic_margin - base_margin

            base_pred = torch.argmax(base_logits, dim=-1)
            ic_pred = torch.argmax(ic_logits, dim=-1)

            valid_float = valid.float()
            denom = valid_float.sum().clamp_min(1.0)

            base_top1_acc = ((base_pred == safe_targets) & valid).float().sum() / denom
            ic_top1_acc = ((ic_pred == safe_targets) & valid).float().sum() / denom

            base_margin_mean = (base_margin * valid_float).sum() / denom
            ic_margin_mean = (ic_margin * valid_float).sum() / denom
            margin_gain_mean = (margin_gain * valid_float).sum() / denom

        return {
            "base_margin": float(base_margin_mean.item()),
            "ic_margin": float(ic_margin_mean.item()),
            "margin_gain": float(margin_gain_mean.item()),
            "base_top1_acc": float(base_top1_acc.item()),
            "ic_top1_acc": float(ic_top1_acc.item()),
        }

    def _compute_margin_loss(self, ic_logits, targets):
        valid = targets != -1
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
        competitor_ids = self._get_strongest_unselected_ids(ic_logits, safe_targets)

        target_logits = self._gather_logits(ic_logits, safe_targets)
        competitor_logits = self._gather_logits(ic_logits, competitor_ids)

        margins = target_logits - competitor_logits
        losses = F.relu(self.config.traj_ic_margin_target - margins)

        valid_float = valid.float()
        denom = valid_float.sum().clamp_min(1.0)
        margin_loss = (losses * valid_float).sum() / denom
        return margin_loss

    def _run_transformer(self, idx, injected_state=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        if injected_state is not None:
            tok_emb = tok_emb + injected_state

        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return x, logits

    def _build_reference_teacher_logits(self, idx, targets):
        """
        Build teacher logits by replacing each position's input with the selected token embedding
        (teacher-forcing), then forward again.
        """
        B, T = idx.shape

        # original embeddings
        tok_emb = self.transformer.wte(idx)

        # replace each position with GT embedding (teacher forcing context edit)
        e_selected = self.transformer.wte(targets.clamp(min=0))

        # only replace valid positions
        mask = (targets != -1).unsqueeze(-1)
        tok_emb_edited = torch.where(mask, e_selected, tok_emb)

        # run transformer
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb_edited + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        teacher_logits = self.lm_head(x)
        return teacher_logits

    def _compute_dist_loss(self, ic_logits, teacher_logits):
        log_probs_ic = F.log_softmax(ic_logits, dim=-1)
        probs_teacher = F.softmax(teacher_logits, dim=-1)

        return F.kl_div(
            log_probs_ic,
            probs_teacher,
            reduction="batchmean"
        )

    def forward(self, idx, targets=None, return_aux=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # inference path: baseline only for now
        if targets is None:
            x, logits = self._run_transformer(idx)
            logits = logits[:, [-1], :]
            loss = None
            if return_aux:
                return logits, loss, {}
            return logits, loss

        # base pass
        h_base, base_logits = self._run_transformer(idx)
        base_loss = self._masked_cross_entropy(base_logits, targets)

        if not self.config.use_traj_ic:
            self.last_traj_ic_stats = {
                "base_loss": float(base_loss.detach().item()),
                "ic_loss": None,
                "traj_loss": None,
                "margin_loss": None,
                "delta_u_norm": None,
                "gate_u_mean": None,
            }
            self.last_probe_stats = {}
            if return_aux:
                return base_logits, base_loss, {}
            return base_logits, base_loss

        valid = targets != -1
        selected_ids = torch.where(valid, targets, torch.zeros_like(targets))
        unselected_ids = self._get_strongest_unselected_ids(base_logits, selected_ids)

        e_selected = self.transformer.wte(selected_ids)
        e_unselected = self.transformer.wte(unselected_ids)

        u_ic, delta_u, gate_u = self.traj_editor(
            h_base, e_selected, e_unselected, alpha=self.config.traj_ic_alpha
        )

        injected_state = torch.zeros_like(h_base)
        injected_state[:, 1:, :] = u_ic[:, :-1, :]  # feed mitigated output at t into t+1

        h_ic, ic_logits = self._run_transformer(idx, injected_state=injected_state)
        ic_loss = self._masked_cross_entropy(ic_logits, targets)
        # ---- teacher distribution ----
        teacher_logits = self._build_reference_teacher_logits(idx, targets)

        # ---- dist loss ----
        dist_loss = self._compute_dist_loss(ic_logits, teacher_logits)

        # simple trajectory consistency surrogate: align next-step hidden state with shifted u_ic
        traj_target = torch.zeros_like(h_base)
        traj_target[:, 1:, :] = u_ic[:, :-1, :]
        traj_loss = ((h_ic - traj_target) ** 2).mean()

        margin_loss = self._compute_margin_loss(ic_logits, targets)

        loss = (
            self.config.traj_ic_lambda_base * base_loss
            + self.config.traj_ic_lambda_ic * ic_loss
            + self.config.traj_ic_lambda_traj * traj_loss
            + self.config.traj_ic_lambda_margin * margin_loss
            + self.config.traj_ic_lambda_dist * dist_loss
        )

        delta_u_norm = delta_u.norm(dim=-1).mean()
        gate_u_mean = gate_u.mean()

        self.last_traj_ic_stats = {
            "base_loss": float(base_loss.detach().item()),
            "ic_loss": float(ic_loss.detach().item()),
            "traj_loss": float(traj_loss.detach().item()),
            "margin_loss": float(margin_loss.detach().item()),
            "delta_u_norm": float(delta_u_norm.detach().item()),
            "gate_u_mean": float(gate_u_mean.detach().item()),
            "dist_loss": float(dist_loss.detach().item()),
        }
        self.last_probe_stats = self._compute_probe_stats(base_logits, ic_logits, targets)

        if return_aux:
            aux = {
                "base_logits": base_logits,
                "ic_logits": ic_logits,
                "selected_ids": selected_ids,
                "unselected_ids": unselected_ids,
                "u_ic": u_ic,
            }
            return ic_logits, loss, aux

        return ic_logits, loss

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
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]

        for k in sd_keys_hf:
            if k not in sd:
                continue
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
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
        # keep baseline generation for now
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
