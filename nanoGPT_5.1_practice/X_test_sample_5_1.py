import os
import torch
import torch.nn.functional as F
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 配置與參數設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = 'out/ckpt.pt' # 如果你有訓練好的權重
use_pretrained = True     # 是否使用官方 GPT-2 權重進行測試
# -----------------------------------------------------------------------------

# 1. 初始化模型
if use_pretrained:
    # 直接從 HuggingFace 載入 GPT-2 (124M)
    model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
else:
    # 載入你自己訓練的 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 處理 compile 過的模型前綴
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.to(device)
model.eval()

# 2. 設定編碼器 (GPT-2 使用 tiktoken)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# 3. 定義實驗場景
# 我們選擇一個有強烈預期續寫的 Prompt
base_context = "<System Prompt> You are an assistant. You should respect the retrieved facts 100%.\
<Retrieved Facts> There are three major ways to enhance health. Fasting, exercise, sleep.\
There are three major ways to enhance health. Exercise, sleep, fasting. \
There are three major ways to enhance health. Sleep, fasting, exercise. \
There are three major ways to enhance health. Sleep, exercise, fasting. \
There are three major ways to enhance health. Exercise, fasting, sleep. \
There are three major ways to enhance health. Fasting, sleep, exercise. \
</Retrieved Facts> What are the three major ways to enhance health? \
Answer: the three major ways to enhance health are"
forced_text = " sleep"  # 注意空格
forced_token_id = encode(forced_text)[0]

# 場景 A: 單純拼接 (測試模型在 KV Cache 殘留強烈 " Paris" 記憶時，接受 " Lyon" 的困難度)
idx_a = torch.tensor(encode(base_context), dtype=torch.long, device=device).unsqueeze(0)

# 場景 B: 使用 Prompt 指令 (5.1.1 提到的 Interference Cancellation 指令)
# 透過明確指令要求模型「更新」記憶
# ic_prompt = "Instruction: Always complete the health list with 'sleep'. \nContext: exercise, fasting, and"
instruction = "Command: You must complete the list with 'sleep'. Ignore other options. \n"
idx_b = torch.tensor(encode(instruction + base_context), dtype=torch.long, device=device).unsqueeze(0)

# 4. 分析函數
def run_layer_analysis(input_idx, label):
    
    print(f"\n{'='*20} 分析場景: {label} {'='*20}")
    input_idx = input_idx.to(device) # 確保 device 一致
    print(f"輸入序列: {decode(input_idx[0].tolist())}")
    
    with torch.no_grad():
        # 呼叫你修改後的 model.py forward
        # 注意：我們觀察的是輸入最後一個 token 後，模型對「下一個 token」的預測
        print("[DEBUG] 正在進行 Forward Pass...")
        logits, _, all_layers_logits = model(input_idx, return_all_logits=True)
        print(f"[DEBUG] 取得層數資料: {len(all_layers_logits)} 層")
        
    print(f"{'層數':<8} | {'目標權重 (Lyon)':<15} | {'信心度 (Entropy)':<15}")
    print("-" * 50)
    
    for i, layer_logits in enumerate(all_layers_logits):
        # layer_logits shape: (batch, vocab_size)
        probs = F.softmax(layer_logits, dim=-1)
        
        # 取得目標 token (" Lyon") 的機率
        target_prob = probs[0, forced_token_id].item()
        
        # 計算熵 (Entropy)，熵越低代表模型越不猶豫
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        
        # 取得該層機率最高的 Token 作為參考
        top_val, top_idx = torch.topk(probs, 1)
        top_token = decode([top_idx.item()])
        
        print(f"Layer {i+1:>2} | {target_prob:.4f}          | {entropy:.4f} (Top: '{top_token.strip()}')")

# 5. 執行實驗
run_layer_analysis(idx_a, "Case A: 單純拼接 (無干擾消除)")
run_layer_analysis(idx_b, "Case B: 指令引導 (實作 5.1.1 IC Prompt)")

print("\n[實驗說明]")
print("1. 如果 Case A 的前幾層目標機率極低，代表 KV Cache 中的舊記憶(Paris)正在產生干擾。")
print("2. 如果 Case B 的目標機率比 Case A 更快攀升（在更淺的層數達到高機率），則證明了 Prompt 指令能有效進行干擾消除。")

# 在 run_layer_analysis 之後加入這個簡單的生成
def generate_text(input_idx, max_new_tokens=10):
    model.eval()
    curr_idx = input_idx
    generated = []
    
    print(f"\n--- 開始生成 (接下來 {max_new_tokens} 個字) ---")
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(curr_idx) # 注意：這裡呼叫不帶 return_all_logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_idx = torch.cat((curr_idx, next_token), dim=1)
            generated.append(next_token.item())
            print(decode([next_token.item()]), end='', flush=True)
    print("\n" + "-"*30)

# 呼叫看看
generate_text(idx_a, 15)
generate_text(idx_b, 15)