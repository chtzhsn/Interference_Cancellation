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
