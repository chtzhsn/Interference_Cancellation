# KV-based Interference Cancellation (IC) – Architecture Overview

---

## 🎯 Goal

Move IC from **hidden-state local correction** to **KV cache rewriting after token selection**, aligning with the original research objective:

> After a token is selected, use hindsight to adjust KV cache so that future attention is more consistent with that decision.

---

# 1️⃣ Baseline Transformer (Control)

```text
Input tokens x_1, ..., x_t
          │
          ▼
Token / Position Embedding
          │
          ▼
Transformer blocks
          │
          ├── produce hidden state h_t
          │
          ├── project to k_t, v_t
          │
          ├── write (k_t, v_t) into KV cache
          │
          ▼
      lm_head(h_t)
          │
          ▼
       Base logits
          │
          ▼
 External token selection
          │
          ▼
 selected token x̂_t
          │
          ▼
 next step attends to original KV cache
```

### 🔍 Problem

* Token is already selected
* KV cache still contains information from **unselected continuations**
* → Causes **interference**

---

# 2️⃣ V-only KV-based IC (Recommended First Version)

```text
Input tokens x_1, ..., x_t
          │
          ▼
Token / Position Embedding
          │
          ▼
Transformer blocks
          │
          ├── produce hidden state h_t
          │
          ├── project to k_t, v_t
          │
          ├── write k_t temporarily
          │
          ▼
      lm_head(h_t)
          │
          ▼
       Base logits
          │
          ▼
 External token selection
          │
          ├── selected token      x̂_t
          └── strongest unselected x̃_t
                    │
                    ▼
      Embedding lookup:
      e_sel = E(x̂_t), e_unsel = E(x̃_t)
                    │
                    ▼
      V-IC module:
      Δv_t = f_v(v_t, e_sel, e_unsel)
                    │
                    ▼
      corrected value:
      v_t' = v_t + α · g_v ⊙ Δv_t
                    │
                    ▼
      write (k_t, v_t') into KV cache
                    │
                    ▼
      next step attends to corrected KV cache
```

---

## 🧠 Interpretation

* **K (keys)** → where to attend
* **V (values)** → what content is retrieved

👉 This version:

* Does NOT change attention weights
* Only modifies retrieved content

---

## ✔ Advantage

* Stable
* Minimal modification
* Closest first step to research goal

---

# 3️⃣ Full KV-based IC (K + V)

```text
Input tokens x_1, ..., x_t
          │
          ▼
Token / Position Embedding
          │
          ▼
Transformer blocks
          │
          ├── produce hidden state h_t
          │
          ├── project to k_t, v_t
          │
          ▼
      lm_head(h_t)
          │
          ▼
       Base logits
          │
          ▼
 External token selection
          │
          ├── selected token      x̂_t
          └── strongest unselected x̃_t
                    │
                    ▼
      Embedding lookup:
      e_sel, e_unsel
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
 K-IC module               V-IC module
 Δk_t = f_k(...)           Δv_t = f_v(...)
         │                     │
         ▼                     ▼
 k_t' = k_t + α_k g_k⊙Δk_t    v_t' = v_t + α_v g_v⊙Δv_t
         └──────────┬──────────┘
                    ▼
        write (k_t', v_t') into KV cache
                    │
                    ▼
      next step attends to corrected KV cache
```

---

## 🧠 Interpretation

* Modify both:

  * **where attention goes (K)**
  * **what it retrieves (V)**

---

## ✔ Closer to theoretical IC

Matches idea:

> suppress unselected token influence while preserving selected token signal

---

# 4️⃣ Hindsight KV Cache Editing (Advanced Version)

```text
At time t:
  base logits → selection gives x̂_t and x̃_t
                         │
                         ▼
        For s in {t-r, ..., t}:
            k_s' = K_ic(k_s, e_sel, e_unsel)
            v_s' = V_ic(v_s, e_sel, e_unsel)
                         │
                         ▼
      overwrite recent KV cache entries
                         │
                         ▼
      step t+1 attends to corrected cache
```

---

## 🧠 Key Idea

* Modify **past memory**
* Not just current step

---

## 🔥 This is closest to original research vision

👉 “Hindsight transformer”
👉 “KV cache editing”

---

# 5️⃣ Architecture Evolution Summary

| Version    | What is modified      | Complexity | Purpose               |
| ---------- | --------------------- | ---------- | --------------------- |
| Baseline   | Nothing               | Low        | Control               |
| V-only IC  | V_t                   | Low        | Safe first step       |
| KV IC      | K_t + V_t             | Medium     | Full local correction |
| Cache edit | K_s, V_s (past steps) | High       | True hindsight        |

---

# 6️⃣ Key Insight

> Interference should be corrected **after token selection**,
> by rewriting KV cache,
> not only by modifying hidden state.

---

# 7️⃣ One-Sentence Summary (for presentation)

> We move interference cancellation from hidden-state perturbation to KV cache rewriting after token selection, enabling hindsight to directly influence future attention and generation.

---
Training Structure
```
INPUT: x_0, x_1, ..., x_T
        │
        ▼
==============================
PASS 1 (Baseline branch)
==============================
        │
        ▼
[Embedding + Position]
        │
        ▼
[Transformer Blocks]
        │
        ▼
z_base (logits)
        │
        ├──► 用來算 base_loss
        │
        └──► 用來產生：
                x_sel(t)   = y_t
                x_unsel(t) = argmax_{j≠y_t} z_base
        │
        ▼
==============================
PASS 2 (IC branch)
==============================
        │
        ▼
[Embedding + Position]
        │
        ▼
[Transformer Blocks + KV IC]
        │
        ▼
z_IC (logits)
        │
        ├──► 用來算 ic_loss
        │
        └──► 用來算 ic accuracy（你畫圖用的）
        
==============================
FINAL OUTPUT（training）
==============================
loss = L_base + L_IC   或其他版本

TRANSFORMER BLOCK:
for each block ℓ:

    input: h_1 ... h_T
        │
        ▼
    [LayerNorm]
        │
        ▼
    [Linear → Q, K, V]
        │
        ▼
    ┌───────────────────────────────┐
    │  IC Module (只在 PASS 2)       │
    │                               │
    │  for each query q:            │
    │      取 (q-1) 的 decision     │
    │      │                        │
    │      ├─ e_sel(q-1)            │
    │      └─ e_unsel(q-1)          │
    │                               │
    │      for each s < q:          │
    │          [K_s, V_s, e_sel, e_unsel]
    │                │
    │                ▼
    │          ΔK, ΔV (MLP)
    │                │
    │                ▼
    │          gate g_K, g_V
    │                │
    │                ▼
    │          K_IC(q,s)
    │          V_IC(q,s)
    │
    │      s = q → 不改
    └───────────────────────────────┘
        │
        ▼
    [Attention]
        │
        ▼
    output o_q
        │
        ▼
    [Residual]
        │
        ▼
    [MLP]
        │
        ▼
    [Residual]
        │
        ▼
    h_q^ℓ

TRAINING IC                  vs        TRUE IC INFERENCE
------------------------------------------------------------
一次吃整段 sequence           一步一步生成 token

z_base / z_IC 被比較          z 決定 token

token = y_t                  token = argmax(z)

KV 是 "每個 query 重算"        KV 是 "真的被覆寫"

沒有記憶累積                 有 persistent memory

IC 不影響輸出                IC 直接控制輸出
```

Inference structure
```
            (一步一步生成)

t = 0
---------------------------------
input: x_0
    │
    ▼
Transformer
    │
    ▼
z_0
    │
    ▼
x_0^* = argmax(z_0)   ← 真正決定 token
    │
    ▼
用 (x_sel^0, x_unsel^0) 改 KV
(K_0 → K_0')

---------------------------------

t = 1
---------------------------------
input: x_0^*, x_1
    │
    ▼
Transformer (用 K_0')
    │
    ▼
z_1
    │
    ▼
x_1^* = argmax(z_1)
    │
    ▼
更新 KV:
K_0' → K_0''
K_1  → K_1'

---------------------------------

t = 2
---------------------------------
input: x_0^*, x_1^*, x_2
    │
    ▼
Transformer (用 K_0'', K_1')
    │
    ▼
z_2
    │
    ▼
x_2^* = argmax(z_2)
    │
    ▼
持續更新 KV
```


