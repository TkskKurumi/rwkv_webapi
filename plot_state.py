from matplotlib import pyplot as plt
from modules.my_rwkv import model, tokenizer, Generator
import torch
import numpy as np
prompt = "Bob: Hi, Alice. How are you doing recently. Alice: I'm fine, everything is going well."
tokens = tokenizer.encode(prompt)
states = []

G = Generator(None, None, model, tokenizer)
last_decoded = ""
for idx, t in enumerate(tokens):
    G = G.feed([t])
    token_prefix = tokens[:idx+1]
    decoded = tokenizer.decode(token_prefix)
    if(prompt.startswith(decoded)):
        char = decoded[len(last_decoded):]
        last_decoded = decoded
        states.append((G.state, char))

for i in range(200):
    token, G = G.sample()
    tokens.append(token)
    decoded = tokenizer.decode(tokens)
    if(decoded[-1] == "\ufffd"):
        token, G = G.sample()
        tokens.append(token)
        decoded = tokenizer.decode(tokens)
    
    char = decoded[len(last_decoded):]
    last_decoded = decoded
    states.append((G.state, char))

diffs = []
diff_abs_means = []

means = []
stds  = []
for idx, i in enumerate(states):
    S, char = i
    if(idx):
        diff = []
        diff_abs_mean = []
        mean = []
        std  = []
        for j in range(0, len(S)):
            _diff = S[j]-states[idx-1][0][j]
            diff.append(_diff)
            
            mean.append(S[j].mean().to(torch.float32).numpy())
            std.append( S[j].std().to(torch.float32).numpy())
            # diff_abs_mean.append(_diff.abs().mean().to(torch.float32).numpy())
            # diff_abs_mean.append(_diff.square().sum().sqrt().to(torch.float32).numpy())
    else:
        diff = [0] * len(S)
        diff_abs_mean = [0] * len(S)

        mean = [S[j].mean().to(torch.float32).numpy() for j in range(0, len(S))]
        std  = [S[j].std().to(torch.float32).numpy()  for j in range(0, len(S))]
    diffs.append(diff)
    diff_abs_means.append(diff_abs_mean)
    means.append(mean)
    stds.append(std)


n_states = len(G.state)
n_layers = n_states//5
n_chars = len(states)
meow = {0: "sx", 1: "aa", 2: "bb", 3: "pp", 4: "sx_ffn"}
plt.figure(figsize=(n_chars, n_layers*2))
for i in range(0, n_layers):
    plt.subplot(n_layers, 1, i+1)
    for j in range(5):
        layer_index = i
        state_name = meow[i%5]
        
        std = [_[i*5+j] for _ in stds]
        mean = [_[i*5+j] for _ in means]
        xs = np.arange(len(states))
        xticks = [char for state, char in states]
        plt.plot(xs, std , label="%02d-%s-std"%(i, meow[j]))
        # plt.plot(xs, mean, label="%02d-%s-m"%(i, meow[j]))
        plt.xticks(xs, xticks)
    plt.legend()
plt.tight_layout()
if(len(last_decoded)>180):
    last_decoded = last_decoded[:180]
plt.savefig("./%s.png"%last_decoded)
print("./%s.png"%last_decoded)


            






