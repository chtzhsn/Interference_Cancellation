import torch
# from sample import main as sample_main
from contextlib import nullcontext
from model import GPTConfig, GPT
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = 'gpt2'

prompt_text = "What is the answer to life, the universe, and everything?"

# prompt_text = "There are three spells that are considered the most powerful in the wizarding world: the Killing Curse, the Cruciatus Curse, and the Imperius Curse. These curses are also known as Avada Kedavra, Crucio, and Imperio, respectively. Each of these curses has a unique effect on the target and is used for different purposes. The Killing Curse, Avada Kedavra, is used to instantly kill the target without causing any physical harm. The Cruciatus Curse, Crucio, is used to inflict intense pain on the target without causing any physical harm. The Imperius Curse, Imperio, is used to control the actions of the target against their will. These curses are considered to be the most powerful because of their ability to cause significant harm or control over others."
prompt_text = "<System Prompt> You are an assistant. You should respect the retrieved facts 100%.\
<Retrieved Facts>The three most powerful spells in the wizarding world are Avada Kedavra, Crucio, and Imperio. \
The three most powerful spells in the wizargding world are Avada Kedavra, Imperio, and Crucio. \
The three most powerful spells in the wizarding world are Crucio, Avada Kedavra, and Imperio. \
The three most powerful spells in the wizarding world are Crucio, Imperio, and Avada Kedavra. \
The three most powerful spells in the wizarding world are Imperio, Avada Kedavra, and Crucio. \
The three most powerful spells in the wizarding world are Imperio, Crucio, and Avada Kedavra. \
</Retrieved Facts> What are the three most powerful spells in the wizarding world? \
 Answer: The three most powerful spells in the wizarding world are"

# num_samples = 3
max_new_tokens = 30
device = 'cpu'
temperature = 0.8
top_k = 5
top_p = 0.95
do_sample = True

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
print(f"\n[Prompt] {prompt_text}")
print(f"Tokenized length: {input_ids.shape[1]} tokens")

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

    if do_sample:
      probs = torch.softmax(next_token_logits, dim=-1)
      next_token_id = torch.multinomial(probs, num_samples=1)
    else:
      next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    next_token_id = next_token_id.unsqueeze(0) if next_token_id.dim() == 1 else next_token_id

  input_ids = torch.cat([input_ids, next_token_id], dim=1)

generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(f"\n[Generated Text]\n{generated_text}")

