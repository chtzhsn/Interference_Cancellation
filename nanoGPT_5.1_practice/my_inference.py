import torch
# from sample import main as sample_main
from contextlib import nullcontext
from model import GPTConfig, GPT
from transformers import GPT2Tokenizer, GPT2LMHeadModel

init_from = 'gpt2'
start = "What is the answer to life, the universe, and everything?"
num_samples = 3
max_new_tokens = 100
device = 'cpu'
# temperature = 0.8
# top_k = 50

tokenizer = GPT2Tokenizer.from_pretrained(init_from)
model = GPT2LMHeadModel.from_pretrained(init_from)
model.to(device)

for k in range(num_samples):
  inputs = tokenizer(start, return_tensors="pt").to(device)
  outputs = model.generate(
    inputs['input_ids'], 
    max_length = len(inputs['input_ids'][0]) + max_new_tokens,
    temperature = 0.8,
    top_k = 50,
    top_p = 0.95,
    do_sample = True,
    pad_token_id = tokenizer.eos_token_id
  )
  text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  print(f"=== Sample {k+1} ===")
  print(text)

# if __name__ == "__main__":
#   sample_main(
#     init_from=init_from,
#     start=start,
#     num_samples=num_samples,
#     max_new_tokens=max_new_tokens,
#     device=device,
#     # temperature=temperature,
#     # top_k=top_k
#   )

