import os
import torch
from tqdm import trange
# os.environ["RWKV_STRATEGY"] = "cuda fp16 *16 -> cpu bf16"
os.environ["RWKV_VOCAB"] = "rwkv_vocab_v20230424"
os.environ["RWKV_STRATEGY"] = "cpu bf16"
from modules.my_rwkv import model, tokenizer, Generator
from modules.initial import init_generator
from modules.lora_strategies import apply_lora, poly, constant
lora_w = torch.load(r'/root/RWKV_Train/save_7B_world/rwkv-30.pth', map_location="cpu")
# lora_w = torch.load(r"/root/RWKV_Train/save_7B_32_ffn/rwkv-14.pth", map_location="cpu")

# ln -sf ~/autodl-tmp/RWKV-4-Raven-7B-v12*.pth model.pth

# ln -sf /root/autodl-tmp/RWKV-4-World-7B-*.pth model.pth
#

for strategy in [
        constant(1),
        poly((0, 1), (0.2, 0.7), (0.8, 0.7), (1, 1))
    ]:

    blend_sum = sum([strategy(i/31) for i in range(32)])/32
    print(blend_sum)

    apply_lora(model, lora_w, strategy, lora_alpha=64, mm_device="cuda:0")
    for task in ["你现在在想什么？", "你是谁？", "什么是逻辑综合？", "最高的山峰是哪一座"]:
        G = init_generator(f"千千：{task}\n菜菜")
        G = G.derive(top_p=0)
        tokens = []
        _ = ""
        print(tokenizer.decode(G.history), end="")
        for i in range(50):
            token, G = G.sample()
            tokens.append(token)
            contents = tokenizer.decode(tokens)
            for i in range(4):
                if(contents[-1] != "\ufffd"): break
                token, G = G.sample()
                tokens.append(token)
                contents = tokenizer.decode(tokens)
            print(contents[len(_):], end="", flush=True)
            _ = contents
        print("="*20)
        # print(tokenizer.decode(G.history))
