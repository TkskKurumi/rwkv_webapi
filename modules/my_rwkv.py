from __future__ import annotations
from tqdm import tqdm
import os
from . import torch_opt
import torch
from dataclasses import dataclass
from types import NoneType
from threading import Lock
import copy
# prepare
model_name = os.environ.get("RWKV_MODEL_PTH", "model.pth")
strategy = os.environ.get("RWKV_STRATEGY", 'cuda fp16')
AVOID_REPEAT = '，。：？！'
os.environ["RWKV_JIT_ON"] = "1"

# some settings must set before import
from rwkv.model import RWKV                          # nopep8
from rwkv.utils import PIPELINE as TokenizerPipeline # nopep8



class Generator:
    
    def __init__(self, out, state, model: RWKV, tokenizer: TokenizerPipeline, temperature=1, top_p=0.2, freq_penalty=0.5, occurrence=None, adjust=None, history=None, state_decay=0):
        self.out = out
        self.state = state
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = freq_penalty
        self.occurrence = occurrence if occurrence is not None else {}
        self.adjust = adjust if adjust is not None else {}
        self.history = history if history is not None else []
        self.state_decay = state_decay
    def derive(self, **kwargs):
        kwa = dict()
        kwa.update(self.__dict__)
        kwa.update(kwargs)
        
        if(kwa["occurrence"] is self.occurrence):
            kwa["occurrence"] = copy.copy(self.occurrence)
        if(kwa["history"] is self.history):
            kwa["history"] = copy.copy(self.history)
        adj = copy.copy(self.adjust)
        for k, v in kwargs.get("adjust", {}).items():
            adj[k] = adj.get(k, 0) + v
        kwa["adjust"] = adj

        return Generator(**kwa)
    def stream(self, n=50, **kwargs):
        if(kwargs):
            nxt = self.derive(**kwargs)
        else:
            nxt = self
        for i in range(n):
            token, nxt = nxt.sample()
            if(token!=0):
                yield token, nxt
            else:
                return
        return
    def feed(self, prompt_or_tokens, slice=2, **kwargs):
        if(isinstance(prompt_or_tokens, str)):
            tokens = self.tokenizer.encode(prompt_or_tokens)
        elif(isinstance(prompt_or_tokens, list)):
            tokens = prompt_or_tokens
        else:
            raise TypeError(type(prompt_or_tokens))
        new_occurrence = copy.copy(self.occurrence)
        for t in tokens:
            new_occurrence[t] = new_occurrence.get(t, 0)+1
        ntokens = len(tokens)
        state = copy.deepcopy(self.state)
        iterator = list(range(0, ntokens, slice))
        if(len(iterator)>10):
            iterator = tqdm(iterator)
        for i in iterator:
            step_tokens = tokens[i:i+slice]
            state0 = state
            out, state = self.model.forward(step_tokens, state)
            if((state0 is not None) and (self.state_decay)):
                decay = self.state_decay**len(step_tokens)
                for idx, i in enumerate(state):
                    state[idx] = state[idx]*(1-decay)+state0[idx]*decay

        newstat = self.derive(out=out, state=state, occurrence=new_occurrence, history=self.history+tokens, **kwargs)
        return newstat
    def sample(self, **kwargs):
        if(self.out is None):
            raise ValueError("generator is not initialized with initial prompt.")
        if(kwargs):
            return self.derive(**kwargs).sample()
        out = copy.deepcopy(self.out)
        for k, v in self.occurrence.items():
            out[k] -= v*self.freq_penalty
        for k, v in self.adjust.items():
            out[k] += v
        token = self.tokenizer.sample_logits(out, temperature=self.temperature, top_p=self.top_p)
        newout, newstate = self.model.forward([token], copy.deepcopy(self.state))
        if((self.state is not None) and (self.state_decay)):
            decay = self.state_decay
            for idx, i in enumerate(newstate):
                newstate[idx] = newstate[idx]*(1-decay)+self.state[idx]*decay
        new_occur = copy.copy(self.occurrence)
        new_occur[token] = new_occur.get(token, 0)+1
        newstat = self.derive(
            out=newout,
            state=newstate,
            occurrence=new_occur,
            history = self.history+[token]
        )
        return token, newstat
        

lora_pth = os.environ.get("LORA_PTH", "")
lora_alpha = os.environ.get("LORA_ALPHA")
if(lora_alpha is not None):
    lora_alpha = float(lora_alpha)
lora_strategy = os.environ.get("LORA_STRATEGY", "constant(1)")

# model = RWKV(model=model_name, strategy=strategy, lora=lora_pth, lora_strategy=lora_strategy, lora_alpha=lora_alpha, lora_mm_device="cuda:0")
model = RWKV(model=model_name, strategy=strategy)
tokenizer = TokenizerPipeline(model, "20B_tokenizer.json")


if(__name__=="__main__"):
    import traceback
    def multi_line_in(p="", cont=""):
        ret = input(p)
        while(ret[-2:]=="\\n"):
            print(cont, end="")
            ret += input()
        return ret.replace("\\n", "\n")
    while(True):
        i = multi_line_in(">> ", ".. ")
        try:
            exec("print("+i+")")
        except:
            try:
                exec(i)
            except:
                traceback.print_exc()
