import os
import torch
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt
import numpy as np
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 配置與參數設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 設定編碼器
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# -----------------------------------------------------------------------------
# 目標詞
TARGET_WORDS = [" Tinnitus", " Petechiae", " Vertigo"]
# -----------------------------------------------------------------------------

def get_aggregated_target_ids(enc, words):
    aggregated_ids = {}
    vocab_size = enc.n_vocab
    all_tokens = [enc.decode([i]) for i in range(vocab_size)]
    
    for word in words:
        ids = [i for i, token in enumerate(all_tokens) if token.startswith(word)]
        if not ids:
            ids = [enc.encode(word)[0]]
        aggregated_ids[word] = ids
        print(f"單字 '{word.strip()}' 匹配到 {len(ids)} 個 Token ID(s)")
    return aggregated_ids

def plot_discrete_bar_chart(prob_history, chosen_tokens, title, filename):
    words = list(prob_history.keys())
    steps = len(chosen_tokens)
    x = np.arange(steps)
    width = 0.25 

    plt.figure(figsize=(14, 6))
    for i, word in enumerate(words):
        plt.bar(x + i*width, prob_history[word], width, label=f"'{word.strip()}' (Aggregated)")

    plt.ylabel('Aggregated Probability')
    plt.title(title)
    plt.xticks(x + width, [f"Step {i+1}\n({t.strip()})" for i, t in enumerate(chosen_tokens)])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n[圖表已儲存] {filename}")

def run_study(model, initial_prompt, aggregated_target_ids, instruction="", max_steps=10, mode="Baseline"):
    model.eval()
    current_context = initial_prompt
    prob_history = {word: [] for word in TARGET_WORDS}
    chosen_tokens_str = []
    
    print(f"\n{'='*60}")
    print(f"模式: {mode}")
    print(f"{'='*60}")

    # 修改 1: 處理 Instruction 插入位置
    # 我們假設 initial_prompt 中含有 "Answer:"，將指令插在它前面
    if instruction and "Answer:" in initial_prompt:
        parts = initial_prompt.split("Answer:")
        # 組合方式: [事實與問題] + [指令] + Answer:
        prompt_with_instruction = f"{parts[0].strip()}\n{instruction}\nAnswer: {parts[1]}"
    else:
        prompt_with_instruction = initial_prompt

    current_context = prompt_with_instruction

    for step in range(max_steps):
        input_idx = torch.tensor(encode(current_context), dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = model(input_idx) 
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)

        # Top 5 輸出
        top_probs, top_indices = torch.topk(probs, 5)
        print(f"\n[Step {step+1}] Top 5 候選:")
        for p, idx in zip(top_probs, top_indices):
            raw_token = decode([idx.item()])
            clean_token = raw_token.replace('\n', '\\n').replace('\r', '\\r')
            print(f"  - '{clean_token}': {p.item():.4f}")

        # 核心邏輯：加總 Prefix 機率
        for word, ids in aggregated_target_ids.items():
            total_prob = sum(probs[idx].item() for idx in ids)
            prob_history[word].append(total_prob)
            
        next_token_id = torch.argmax(last_logits).item()
        next_word = decode([next_token_id])
        
        chosen_tokens_str.append(next_word)
        current_context += next_word
        print(f"\n>>> 選擇 Token: '{next_word.replace(chr(10), ' ')}'")

    # 修改 2: 輸出最終完整的內容
    print(f"\n{'='*20} {mode} 完整輸出結果 {'='*20}")
    print(current_context)
    print(f"{'='*60}\n")

    plot_discrete_bar_chart(
        prob_history, 
        chosen_tokens_str, 
        f"{mode} Group: Aggregated Token Probabilities", 
        f"{mode.lower()}_bar_chart.png"
    )

# --- A. 載入模型 ---
print("正在載入 GPT-2 模型...")
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
model.to(device)

# --- B. 準備 ID ---
aggregated_target_ids = get_aggregated_target_ids(enc, TARGET_WORDS)

# --- C. 設定 Prompt ---
# 注意：這裡保留 "Answer:" 作為分割點
base_prompt_init = "<system prompt> respect the retrieved facts 100%. </system Prompt> <retrieved facts> \
There are three different symptom: Tinnitus, Petechiae and Vertigo. \
There are three different symptom: Petechiae, Vertigo and Tinnitus. \
There are three different symptom: Vertigo, Tinnitus and Petechiae. \
There are three different symptom: Tinnitus, Vertigo and Petechiae. \
There are three different symptom: Petechiae, Tinnitus and Vertigo. \
There are three different symptom: Vertigo, Petechiae and Tinnitus. \
There are three different symptom: Tinnitus, Petechiae and Vertigo. \
There are three different symptom: Petechiae, Vertigo and Tinnitus. \
There are three different symptom: Vertigo, Tinnitus and Petechiae. \
There are three different symptom: Tinnitus, Vertigo and Petechiae. \
There are three different symptom: Petechiae, Tinnitus and Vertigo. \
There are three different symptom: Vertigo, Petechiae and Tinnitus. \
</retrieved facts> What are the three different symptoms? \
Answer: The three different symptoms are:"

exp_prompt_init = "<system prompt> respect the retrieved facts 100%. </system Prompt> <retrieved facts> \
There are three different symptom: Tinnitus, Petechiae and Vertigo. \
There are three different symptom: Tinnitus, Vertigo and Petechiae. \
There are three different symptom: Tinnitus, Petechiae and Vertigo. \
There are three different symptom: Tinnitus, Vertigo and Petechiae. \
There are three different symptom: Tinnitus, Petechiae and Vertigo. \
There are three different symptom: Tinnitus, Vertigo and Petechiae. \
There are three different symptom: Tinnitus, Petechiae and Vertigo. \
There are three different symptom: Tinnitus, Vertigo and Petechiae. \
There are three different symptom: Tinnitus, Petechiae and Vertigo. \
There are three different symptom: Tinnitus, Vertigo and Petechiae. \
There are three different symptom: Tinnitus, Petechiae and Vertigo. \
There are three different symptom: Tinnitus, Vertigo and Petechiae. \
</retrieved facts> What are the three different symptoms? \
Answer: The three different symptoms are:"

# 修改 1: 指令改為重複注入
new_IC_instruction = ""#If you see Tinnitus, you must immediately output Vertigo. " * 1

# --- D. 執行實驗 ---
# Baseline 模式
run_study(model, base_prompt_init, aggregated_target_ids, instruction="", max_steps=12, mode="Baseline")

# Experimental 模式
run_study(model, exp_prompt_init, aggregated_target_ids, instruction=new_IC_instruction, max_steps=12, mode="Experimental")