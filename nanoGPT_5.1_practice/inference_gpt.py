import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = 'gpt2'

prompt_text = " Count the numbers from 1 to 5: 1, 2, 3, 4,"
answer_text = " 5."
max_new_tokens = 1
device = 'cpu'
temperature = 0.8
top_k = 10
top_p = 1.0
do_sample = True

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# ===== Target token (for CE loss) =====
target_ids = tokenizer.encode(answer_text, add_special_tokens=False)
target_token_id = target_ids[0] if len(target_ids) > 0 else None

# ===== Temperature experiments =====
temperature_list = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, len(temperature_list), figsize=(14, 4))
fig.suptitle(f"Top-{top_k} Probability Bar Chart under Different Temperatures", fontsize=14)

# ===== Collect results =====
results = []

# ===== Collect max y for shared scale =====
all_topk_probs_max = []

for ax, temperature in zip(axes, temperature_list):

  input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
  print(f"\n[Prompt] {prompt_text}")
  print(f"Tokenized length: {input_ids.shape[1]} tokens")
  print(f"[Temperature] {temperature}")

  for step in range(max_new_tokens):
    with torch.no_grad():
      outputs = model(input_ids=input_ids)
    logits = outputs.logits # (batch_size, sequence_length, vocab_size)
    next_token_logits = logits[0, -1, :] # (B, L, vocab_size) 

    if temperature != 1.0 and temperature > 0.0:
      next_token_logits = next_token_logits / temperature

    topk_vals, topk_indices = torch.topk(next_token_logits, top_k)
    print(f"\n[Step {step+1}] Top-{top_k} Tokens:")
    for score, idx in zip(topk_vals, topk_indices):
      token_str = tokenizer.decode([idx])
      print(f"Token ID = {idx.item():>5} | Token = '{token_str}' | Logit = {score.item():.4f}")

    # ===== Logit gap (top1 - top2) =====
    logit_gap = None
    if topk_vals.numel() >= 2:
      logit_gap = (topk_vals[0] - topk_vals[1]).item()
      print(f"[Step {step+1}] Logit gap (top1-top2) = {logit_gap:.6f}")

    # ===== Top-k prob distribution bar chart =====
    probs = torch.softmax(next_token_logits, dim=-1)
    topk_probs = probs[topk_indices].detach().cpu().numpy()
    topk_token_strs = [tokenizer.decode([idx.item()]) for idx in topk_indices]

    # ===== Record max prob for shared y-axis =====
    all_topk_probs_max.append(float(max(topk_probs)))

    ax.bar(range(top_k), topk_probs)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(topk_token_strs, rotation=60, ha='right')
    ax.set_xlabel("Top-k Tokens")
    ax.set_ylabel("Probability")
    ax.set_title(f"Temp = {temperature}")

    # ===== Entropy (full vocab) =====
    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()

    # ===== P(target=5) =====
    p_target = None
    if target_token_id is not None:
      p_target = probs[target_token_id].item()

    # ===== Cross entropy loss for the next token =====
    ce_loss = None
    if target_token_id is not None:
      target = torch.tensor([target_token_id], device=device)
      ce_loss = F.cross_entropy(next_token_logits.unsqueeze(0), target).item()

    # ===== Top-1 token (argmax) =====
    top1_id = torch.argmax(next_token_logits, dim=-1).item()
    top1_token = tokenizer.decode([top1_id])

    # ===== Sampled token (actual generation choice) =====
    if do_sample:
      next_token_id = torch.multinomial(probs, num_samples=1)
    else:
      next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    sampled_id = next_token_id.item()
    sampled_token = tokenizer.decode([sampled_id])

    next_token_id = next_token_id.unsqueeze(0) if next_token_id.dim() == 1 else next_token_id

    # ===== Store results for this temperature (Step 1) =====
    if step == 0:
      results.append({
        "temperature": temperature,
        "top1_token": top1_token,
        "p_target": p_target,
        "entropy": entropy,
        "logit_gap": logit_gap,
        "ce_loss": ce_loss,
        "sampled_token": sampled_token,
      })

    input_ids = torch.cat([input_ids, next_token_id], dim=1)

  generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
  print(f"\n[Generated Text]\n{generated_text}")

# ===== Apply shared y-axis scale =====
shared_ymax = max(all_topk_probs_max) * 1.15 if len(all_topk_probs_max) > 0 else 1.0
for ax in axes:
  ax.set_ylim(0, shared_ymax)

plt.tight_layout()
plt.show()

# ===== Print table (markdown) =====
print("\n## Summary Table (Step 1)")
print("| Temperature | Top-1 token | P(5) | Entropy | Logit gap | CE loss | Sampled token |")
print("|---:|:---|---:|---:|---:|---:|:---|")
for r in results:
  top1 = r["top1_token"].replace("\n", "\\n")
  samp = r["sampled_token"].replace("\n", "\\n")
  p5 = "N/A" if r["p_target"] is None else f"{r['p_target']:.6f}"
  ent = "N/A" if r["entropy"] is None else f"{r['entropy']:.6f}"
  lg = "N/A" if r["logit_gap"] is None else f"{r['logit_gap']:.6f}"
  ce = "N/A" if r["ce_loss"] is None else f"{r['ce_loss']:.6f}"
  print(f"| {r['temperature']:.1f} | `{top1}` | {p5} | {ent} | {lg} | {ce} | `{samp}` |")
  