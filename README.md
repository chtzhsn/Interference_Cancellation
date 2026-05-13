# Interference Cancellation Project Progress Report

[My code](https://github.com/chtzhsn/Interference_Cancellation)

## 1. Current Progress

### 1.1 KV-based Interference Cancellation Framework

The current work focuses on directly modifying Transformer attention memory through Key/Value intervention:

$$K \rightarrow K', \qquad V \rightarrow V'$$

For each token position $t$, the model first computes baseline logits: $z_t^{base}$

Then the ground-truth token is used as the selected token $x_{\text{sel}}^{(t)} = y_t$

The unselected token is chosen as the highest-logit incorrect token $x_{\text{unsel}}^{(t)}=\arg\max_{j \neq y_t}z_{t,j}^{base}$

For $s < t$, the IC module generates KV corrections:

$$\Delta K_s(t) = f_K(e_{\text{sel}}, e_{\text{unsel}})$$

$$\Delta V_s(t) = f_V(e_{\text{sel}}, e_{\text{unsel}})$$

and updates:

$$K'_s(t) = K_t + \lambda_K g_K \odot \Delta K_s(t)$$

$$V'_s(t) = V_t + \lambda_V g_V \odot \Delta V_s(t)$$

where:
- $e_{\text{sel}}$: embedding of selected token
- $e_{\text{unsel}}$: embedding of competing token
- $g_K, g_V$: learned gates
- $\lambda_K, \lambda_V$: scale

---

The modified KV pairs are then reused by the second-pass attention to produce $z_t^{IC}$

Let batch size = $B$, sequence length = $T$, vocabulary size = $V$

Ground-truth token: $y_{b,t}\in\{1,\dots,V\}$

Base-model logits: $z^{\text{base}}_{b,t}\in\mathbb{R}^V$

IC-model logits: $z^{\text{IC}}_{b,t}\in\mathbb{R}^V$

where $z^{\text{base}}_{b,t,j}$ denotes the base-model logit of token $j$ at batch index $b$ and timestep $t$.

---

#### Base Top-1 Accuracy

Base prediction: $\hat y^{\text{base}}_{b,t}=\arg\max_j z^{\text{base}}_{b,t,j}$

Base correctness indicator: $a^{\text{base}}_{b,t}=\mathbf{1}\left[\hat y^{\text{base}}_{b,t}=y_{b,t}\right]$

where $a^{\text{base}}_{b,t}\in\{0,1\}$

Base top-1 accuracy: $\text{base_top1}=\frac{1}{BT}\sum_{b=1}^B\sum_{t=1}^Ta^{\text{base}}_{b,t}$

---

#### IC Top-1 Accuracy

IC prediction: $\hat y^{\text{IC}}_{b,t}=\arg\max_j z^{\text{IC}}_{b,t,j}$

IC correctness indicator: $a^{\text{IC}}_{b,t}=\mathbf{1}\left[\hat y^{\text{IC}}_{b,t}=y_{b,t}\right]$

IC top-1 accuracy: $\text{ic_top1}=\frac{1}{BT}\sum_{b=1}^B\sum_{t=1}^Ta^{\text{IC}}_{b,t}$

---

Accuracy Gap


$$\Delta_{\text{IC}}=\text{ic_top1}-\text{base_top1} =\frac{1}{BT}\sum_{b,t}\left(a^{\text{IC}}_{b,t}-a^{\text{base}}_{b,t}\right)$$

---

$$a^{\text{IC}}_{b,t}-a^{\text{base}}_{b,t}\in\{-1,0,+1\}$$

- $+1$:  IC corrects a previously wrong prediction
- $0$:  no change
- $-1$:  IC breaks a previously correct prediction

Therefore, $\Delta_{\text{IC}}>0$ means that IC fixes more tokens than it harms on average.

---

### 1.2 Joint Base + IC Loss

$$L=\lambda_{\text{base}} CE(z^{base}, y)+\lambda_{\text{IC}} CE(z^{IC}, y)$$

Purpose:
- stabilize training
- avoid divergence from baseline branch
- 

Observation:

| ![image](https://hackmd.io/_uploads/BJJKAxk1Ge.png) | ![image](https://hackmd.io/_uploads/Hy9tRg1JMx.png) |
| -------- | -------- | 
| ![image](https://hackmd.io/_uploads/HJl5AxkJfe.png)    | ![image](https://hackmd.io/_uploads/ryH9CxyJMe.png)    | 
|![image](https://hackmd.io/_uploads/ryoc0ly1Mx.png) | ![image](https://hackmd.io/_uploads/S1xiCl1yzg.png)| 

- IC accuracy 有微幅超越對照組的趨勢
- modification of V' is accepted stably, while K' is rejected.


---

### 1.3 IC Loss Variants

Several IC loss formulations have been implemented and compared.

---

#### IC-only Loss

$$L=CE(z^{IC}, y)$$

$$K' = K + \Delta K, V' = V + \Delta V$$

Purpose: force IC branch to fully take responsibility

Observation:

| ![image](https://hackmd.io/_uploads/Sy6Xe-J1zg.png)|![image](https://hackmd.io/_uploads/BJhVlZkyGe.png)| 
| -------- | -------- |
|![image](https://hackmd.io/_uploads/BkErlZJkMe.png)| ![image](https://hackmd.io/_uploads/BynreZ1Jfg.png) |
|![image](https://hackmd.io/_uploads/r178gZ11Mg.png)|![image](https://hackmd.io/_uploads/By9LxWykGg.png) |

- larger KV correction magnitude
- clearer IC-vs-baseline differences (stronger IC behavior)
- similarly, V' accepted and K' rejected.

##### KV Ablation Experiments

Two other intervention types were compared: only modify either K or V

###### V-only IC

$$V' = V + \Delta V, \qquad K'=K$$

Observation:
|![image](https://hackmd.io/_uploads/SJlrVbkkfl.png) | ![image](https://hackmd.io/_uploads/rkjH4-kJze.png)| 
| -------- | -------- | 
| ![image](https://hackmd.io/_uploads/SJMIEW11zx.png)  | ![image](https://hackmd.io/_uploads/Bkh84W1kGe.png) | 
| ![image](https://hackmd.io/_uploads/H1zDEb1yfx.png) | ![image](https://hackmd.io/_uploads/BJuwN-1kfx.png) |

- IC's accuracy 明顯上升趨勢
- V' is stably accepted by model (gate_v)
- currently the most stable setting


###### K-only IC

$$K' = K + \Delta K, \qquad V'=V$$

Observation:
| ![image](https://hackmd.io/_uploads/SyvlE-JJzl.png)| ![image](https://hackmd.io/_uploads/ryRe4bJJMx.png)| 
| -------- | -------- | 
| ![image](https://hackmd.io/_uploads/r1BWNby1Ge.png) |![image](https://hackmd.io/_uploads/H1TZN-11fe.png)| 
| ![image](https://hackmd.io/_uploads/rkXMEWkkfg.png) | ![image](https://hackmd.io/_uploads/ByxXN-kyfe.png) | 
- IC effect exists
- relatively unstable
- K' slowly rejected by model


<!-- ---



### 1.4 Margin IC Loss

$$L=CE(z^{IC}, y)+\lambda\max(0, m - (z_y^{IC} - z_{unsel}^{IC}))$$

where:
- $z_y^{IC}$: correct-token logit
- $z_{unsel}^{IC}$: competing-token logit
- m = 0.5

Purpose: explicitly enlarge logit margin

Observation:


| ![image](https://hackmd.io/_uploads/B1_pMWk1ze.png)| ![image](https://hackmd.io/_uploads/r1RTfW11fx.png)| 
| -------- | -------- | 
| ![image](https://hackmd.io/_uploads/B1VRzZkyzl.png) | ![image](https://hackmd.io/_uploads/rJ5AzWJkMl.png) | 
|![image](https://hackmd.io/_uploads/rJly7b1Jfg.png) |![image](https://hackmd.io/_uploads/H1UJm-k1Ge.png) | 


- accuracy improvement is unstable
- K' accepted, V' rejected. -->

---

<!-- ## Conditional IC Loss

IC is only activated under specific conditions:

$$\mathbf{1}[condition]$$

Examples tested:
- activate IC only when baseline is wrong
- activate IC only on low-confidence tokens

Observation:
| ![image](https://hackmd.io/_uploads/HJYKfZkyzx.png)| ![image](https://hackmd.io/_uploads/HyW5MWyyGl.png)| 
| -------- | -------- | 
| ![image](https://hackmd.io/_uploads/r1d9z-kyMe.png)    | ![image](https://hackmd.io/_uploads/ryloGZy1Gl.png) | 
| ![image](https://hackmd.io/_uploads/S1_iMZy1Mg.png)| ![image](https://hackmd.io/_uploads/BJxnzW11Mx.png)|

- highly sensitive to condition design
- optimization stability depends strongly on trigger rule -->


---

### 1.5 Confidence-Bucket Analysis



---

<!-- # 1.5 Experiment Summary (v3.1 ~ v3.10)

| Version | Main Idea | Observation |
|---|---|---|
| v3.1 | Initial trajectory-wide KV IC | IC effect observable but weak |
| v3.2 | Longer trajectory-wide KV IC | Larger KV correction trends appear |
| v3.3 | IC-only loss | IC branch becomes significantly more active |
| v3.4 | Conditional IC | Strong dependence on trigger condition |
| v3.5 | Margin IC | Margin increases but accuracy unstable |
| v3.6 | Trajectory-wide IC refinement | More stable trajectory behavior |
| v3.7 | V-only + IC-only | Most stable low-confidence improvement |
| v3.8 | K-only + IC-only | Stronger fluctuations and instability |
| v3.9 | V-only + confidence bucket analysis | IC mainly improves low-confidence tokens |
| v3.10 | Joint KV + confidence bucket analysis | Stronger gain potential with higher instability | -->

---

# 2. Current Limitation

All current experiments are still conducted under:
- small-model settings
- Tiny Shakespeare dataset
- CPU-only environment

Therefore:
- current results should be viewed as proof-of-concept observations
- large-scale validation has not yet been completed

---

# 3. Required Support and Future Work Allocation

A new team member has joined the project to help accelerate large-scale experimentation and validation.

---

## My Responsibilities

I will continue focusing on:
- IC architecture design
- KV intervention mechanisms
- loss-function design
- confidence-aware IC strategies
- theoretical analysis
- debugging and experiment planning

---

## New Team Member Responsibilities

The new member will mainly assist with:
- large-model training
- validation/testing pipeline construction
- experiment scaling
- reproducibility evaluation
- resource coordination

Future targets include:
- larger GPT-scale models
- longer-context evaluation
- validation on larger datasets
- systematic baseline comparison


<!-- ---

# 4. Overall Direction

The project is currently transitioning from:
- small-scale proof-of-concept IC experiments

toward:
- scalable Transformer-level IC evaluation.

The current priority is:
1. stabilize IC optimization
2. improve low-confidence-token correction
3. validate scalability on larger models and datasets
 -->

---

# 4. Project Goal

Current long-term target: AAAI 2027 submission 
- deadline: 7/27/2026
![image](https://hackmd.io/_uploads/ByMaF1byfl.png)


Main direction:
- Transformer KV intervention
- confidence-aware interference cancellation


The current objective is not only to improve accuracy, but also to understand "when IC helps, and why".

---

# 5. Next-stage Research Directions

## 5.1 Confidence-aware IC

A confidence-bucket analysis pipeline was implemented to study how IC behaves under different baseline confidence regimes.

---

#### Baseline Confidence Definition

For token position $i$, the baseline logits = $z_i^{base} \in \mathbb{R}^{V}$, where $V$ is the vocabulary size.

The baseline softmax probability is:

$$p_{i,j}^{base}=\frac{\exp(z_{i,j}^{base})}{\sum_{k=1}^{V}\exp(z_{i,k}^{base})}$$

The baseline confidence for the ground-truth token is $p_{i,y_i}^{base}$

where:
- $y_i$ is the ground-truth token
- $p_{i,y_i}^{base}$ measures how confident the baseline model is on the correct answer

---

#### Confidence Bucket Partition

Tokens are partitioned into three confidence sets:

High-confidence set: $\mathcal H=\{i \mid p_{i,y_i}^{base} > 0.7\} ,\qquad n_H = |\mathcal H|$


Medium-confidence set: $\mathcal M=\{i \mid 0.3 < p_{i,y_i}^{base} \le 0.7\} ,\qquad n_M = |\mathcal M|$


Low-confidence set: $\mathcal L=\{i \mid p_{i,y_i}^{base} \le 0.3\} ,\qquad n_L = |\mathcal L|$

---

#### Baseline / IC Correctness

For each token position,

Baseline correctness: $a_i^{base}=\mathbf 1\left[\arg\max_j z_{i,j}^{base}=y_i\right]$

IC correctness: $a_i^{IC}=\mathbf 1\left[\arg\max_j z_{i,j}^{IC}=y_i\right]$

where $a_i^{base}, a_i^{IC} \in \{0,1\}$

---

#### Bucket-wise IC Improvement

The IC improvement for each confidence bucket is defined as:

High-confidence improvement: $\Delta_H=\frac1{n_H}\sum_{i\in\mathcal H}(a_i^{IC}-a_i^{base})$

Medium-confidence improvement: $\Delta_M=\frac1{n_M}\sum_{i\in\mathcal M}(a_i^{IC}-a_i^{base})$

Low-confidence improvement: $\Delta_L=\frac1{n_L}\sum_{i\in\mathcal L}(a_i^{IC}-a_i^{base})$

where $a_i^{IC}-a_i^{base}\in\{-1,0,+1\}$ has the following interpretation:

| Value | Meaning |
|---|---|
| $+1$ | baseline incorrect, IC correct |
| $0$ | both predictions identical |
| $-1$ | baseline correct, IC incorrect |

Thus $\Delta_L > 0$ means IC improves low-confidence tokens on average.

---

#### Overall IC Improvement Decomposition

The total IC accuracy gain is:

$$\Delta_{total}=Acc_{IC}-Acc_{base}$$

which can be decomposed as:

$$\Delta_{total}=\frac{n_H}{N}\Delta_H+\frac{n_M}{N}\Delta_M+\frac{n_L}{N}\Delta_L$$

where $N = n_H+n_M+n_L$
Define contribution $C_{(H,M,L)}=\frac{n_{(H,M,L)}}{N}\Delta_{(H,M,L)}$

This decomposition allows us to identify:
- which confidence regime contributes most to the overall IC improvement
- whether IC mainly helps uncertain tokens or already-confident predictions

<!-- ---

#### Current Observation

##### K and V
| ![image](https://hackmd.io/_uploads/HkGLnzJ1Ml.png)| ![image](https://hackmd.io/_uploads/BkY83zJJGe.png) | ![image](https://hackmd.io/_uploads/HkgPMQyyzl.png)|
| -------- | -------- | -------- |

##### V only

| ![image](https://hackmd.io/_uploads/BkIXjGJJze.png)| ![image](https://hackmd.io/_uploads/HyxBszy1fl.png) | ![image](https://hackmd.io/_uploads/H1QLzQykfl.png)|
| -------- | -------- | -------- |

Current experimental results show $\Delta_L > 0$ while $\Delta_H \approx 0$ and medium-confidence tokens may occasionally degrade.

This suggests:
- IC mainly improves uncertain predictions (low confidence)
- IC behaves more like an uncertainty-correction mechanism rather than a uniform performance-improvement mechanism -->

Next step:
- observe IC performance between different confidence
- confidence-conditioned KV gating: $g_{conf}(t) = \mathbf 1\left[p_y(t) < \tau \right]$ or $g_{conf}(t) = 1 - p_y(t)$




---

## 5.2 Large-model Validation

We want to prove low-confidence improvement remain observable at scale.

Next-stage experiments:
- larger GPT models
- longer context windows
- validation/testing evaluation
- multi-seed reproducibility

---

# 6. Short-term Milestones

## Stage 1 (Current)
- complete KV / K-only / V-only ablations
- complete confidence-bucket analysis
- compare different IC losses

## Stage 2
- migrate to larger GPT models
- establish validation/testing pipeline
- evaluate reproducibility
- mathtematically prove why IC works

## Stage 3
- (TBD)
<!-- - study autoregressive KV-cache intervention -->