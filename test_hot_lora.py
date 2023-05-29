import os
import torch
from tqdm import trange
os.environ["RWKV_STRATEGY"] = "cpu bf16"
from modules.my_rwkv import model, tokenizer, Generator
from modules.initial import init_generator
from modules.lora_strategies import apply_lora, poly, constant
lora_w = torch.load(r'/root/RWKV_Train/save_7B_32_ffn/rwkv-8.pth')

for strategy in [constant(0), constant(1), constant(2), poly((0, 1), (0.5, 0), (1, 1))]:

    apply_lora(model, lora_w, strategy, lora_alpha=64, mm_device="cuda:0")

    G = init_generator("千千：你是谁？\n菜菜")
    tokens = []
    for i in trange(50):
        token, G = G.sample()
        tokens.append(token)
        contents = tokenizer.decode(tokens)
        if(contents[-1] == "\ufffd"):
            token, G = G.sample()
            tokens.append(token)
            contents = tokenizer.decode(tokens)
    print(tokenizer.decode(G.history))