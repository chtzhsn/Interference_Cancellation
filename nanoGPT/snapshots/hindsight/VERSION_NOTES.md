# NanoGPT Interference Cancellation — Version Table

## Overview

This document records all experiment versions of the interference cancellation (IC) module, including:

* Code differences
* Training commands
* Expected behaviors
* Research insights

This enables full reproducibility and clear comparison across versions.

---

## Version Summary

| Version | Name           | Key Idea                               | Selected Token        | Probe | Margin Loss | Behavior                            |
| ------- | -------------- | -------------------------------------- | --------------------- | ----- | ----------- | ----------------------------------- |
| v0      | Baseline       | Original nanoGPT                       | N/A                   | ❌     | ❌           | Standard LM                         |
| v1      | Naive IC       | IC with ground-truth selection         | `targets`             | ❌     | ❌           | Cheating / overwrite                |
| v2      | Constrained IC | Prediction-based + conservative update | `argmax(base_logits)` | ❌     | ❌           | Stable but weak                     |
| v3      | Probe IC       | Add interpretability metrics           | `argmax(base_logits)` | ✅     | ❌           | Reveals wrong direction             |
| v4      | Margin IC      | Add supervision for separation         | `argmax(base_logits)` | ✅     | ✅           | Correct direction, no accuracy gain |

---

## Common Workflow

### Step 1 — Copy version code

```bash
cp snapshots/code_versions/model_vX_xxx.py nanoGPT/model.py
cp snapshots/code_versions/train_vX_xxx.py nanoGPT/train.py
```

### Step 2 — Run training

```bash
cd nanoGPT
python train.py ... > ../logs/xxx.txt
```

### Step 3 — Plot results

```bash
cd ..
python plot_training_log.py logs/xxx.txt --outdir plots/xxx
```

---

## v0 — Baseline

### Description

* Original nanoGPT model
* No IC module

### Command

```bash
cd nanoGPT
python train.py config/train_shakespeare_char.py \
--device=cpu --compile=False --use_ic=False \
> ../logs/v0_baseline.txt
```

### Expected Results

* Best validation loss ≈ **2.44**
* No IC-related metrics

---

## v1 — Naive IC

### Description

* Uses ground-truth token as selected token
* IC module can directly reconstruct target

### Key Implementation

```python
selected_ids = targets
```

### Command

```bash
cd nanoGPT
python train.py config/train_shakespeare_char.py \
--device=cpu --compile=False --use_ic=True \
--ic_lambda_base=0.5 --ic_lambda_mit=1.0 \
> ../logs/v1_naive_ic.txt
```

### Expected Behavior

* Extremely low `mit_loss`
* Large `delta_norm`
* Gate saturation
* ❗ **Cheating (latent overwrite)**

---

## v2 — Constrained IC

### Description

* Prevent cheating
* Use model prediction instead of ground truth
* Apply conservative correction

### Key Implementation

```python
selected_ids = argmax(base_logits)
gate = 0.5 * gate
ic_alpha = 0.1
```

### Command

```bash
cd nanoGPT
python train.py config/train_shakespeare_char.py \
--device=cpu --compile=False --use_ic=True \
--ic_lambda_base=1.0 --ic_lambda_mit=0.3 \
> ../logs/v2_constrained_ic.txt
```

### Expected Behavior

* No cheating
* IC effect weak
* Possible degradation vs baseline

---

## v3 — Probe IC

### Description

* Adds interpretability metrics

### Added Metrics

* `base_margin`
* `mit_margin`
* `margin_gain`
* `base_top1`
* `mit_top1`

### Command

```bash
cd nanoGPT
python train.py config/train_shakespeare_char.py \
--device=cpu --compile=False --use_ic=True \
--ic_lambda_base=1.0 --ic_lambda_mit=0.3 \
> ../logs/v3_probe_ic.txt
```

### Expected Behavior

* `margin_gain < 0`
* ❗ IC is moving in the **wrong direction**

---

## v4 — Margin IC

### Description

* Adds explicit supervision via margin loss
* Forces separation between target and competitor

### Key Idea

```python
margin_loss = max(0, margin_target - (target - competitor))
```

### Command

```bash
cd nanoGPT
python train.py config/train_shakespeare_char.py \
--device=cpu --compile=False --use_ic=True \
--ic_lambda_base=1.0 --ic_lambda_mit=0.3 \
--ic_lambda_margin=0.5 --ic_margin_target=0.2 \
> ../logs/v4_margin_ic.txt
```

### Expected Behavior

* `margin_gain > 0`
* `mit_margin > base_margin`
* ❗ Accuracy may **not improve**

---

## Key Research Insight

Across experiments:

1. Naive IC learns trivial reconstruction (cheating)
2. Constrained IC removes cheating but lacks direction
3. Probe reveals IC is not suppressing interference
4. Margin supervision corrects direction
5. However:

> **Interference suppression ≠ better prediction**

---

## Recommended Experiment Order

To reproduce results clearly:

1. v0 → Baseline
2. v1 → Show cheating
3. v2 → Fix cheating
4. v3 → Diagnose failure
5. v4 → Fix direction

---

## Notes

* Always store logs in `logs/`
* Always generate plots in `plots/`
* Never overwrite working version without snapshot

---

## Future Work

* Synthetic branching dataset
* Multi-path prediction evaluation
* KV-space interference modeling

---
