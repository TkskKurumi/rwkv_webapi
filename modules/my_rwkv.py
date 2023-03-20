from __future__ import annotations

import os
from . import torch_opt
import torch
from dataclasses import dataclass
from types import NoneType
from threading import Lock
import copy
# prepare
model_name = os.environ.get("RWKV_MODEL_PTH", "model.pth")
strategy = os.environ.get("RWKV_STRATEGY", 'cuda fp16i8 *10 -> cuda fp16')
AVOID_REPEAT = '，。：？！'
os.environ["RWKV_JIT_ON"] = "1"

# some settings must set before import
from rwkv.model import RWKV                          # nopep8
from rwkv.utils import PIPELINE as TokenizerPipeline # nopep8



class Generator:
    
    def __init__(self, out, state, model: RWKV, tokenizer: TokenizerPipeline, temperature=1, top_p=0.2, freq_penalty=0.5, occurrence=None, adjust=None):
        self.out = out
        self.state = state
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = freq_penalty
        self.occurrence = occurrence if occurrence is not None else {}
        self.adjust = adjust if adjust is not None else {}
    def derive(self, **kwargs):
        kwa = dict()
        kwa.update(self.__dict__)
        kwa.update(kwargs)
        if(kwa["occurrence"] is self.occurrence):
            kwa["occurrence"] = copy.copy(self.occurrence)
        # assert kwa["occurrence"] is not self.occurrence, "Copying dict occurence, it'll be shared"
        return Generator(**kwa)
    def stream(self, n=50, **kwargs):
        if(kwargs):
            nxt = self.derive(**kwargs)
        else:
            nxt = self
        for i in range(n):
            token, nxt = nxt.sample()
            yield token, nxt
        return
    def feed(self, prompt_or_tokens, slice=4, **kwargs):
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
        for i in range(0, ntokens, slice):
            step_tokens = tokens[i:i+slice]
            out, state = self.model.forward(step_tokens, state)
        newstat = self.derive(out=out, state=state, occurrence=new_occurrence, **kwargs)
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
            out[k] -= v
        token = self.tokenizer.sample_logits(out, temperature=self.temperature, top_p=self.top_p)
        newout, newstate = self.model.forward([token], copy.deepcopy(self.state))
        new_occur = copy.copy(self.occurrence)
        new_occur[token] = new_occur.get(token, 0)+1
        newstat = self.derive(out=newout, state=newstate, occurrence=new_occur)
        return token, newstat
        

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
