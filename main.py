from modules.initial import init_english_qa, init_generator, init_neko_a, tokenizer, Generator
from modules.my_rwkv import USING_LORA
from modules.lora_strategies import get_fstrategy
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from tqdm import trange
import copy, json
import torch, time
import numpy as np
from typing import List
from uuid import uuid4
from threading import Lock
from types import NoneType
from pydantic import BaseModel

sanity_check = init_generator("Sanity Check")

base_generators = {
}
generators = {
}
G_time = {}

def get_status(key):
    G_time[key] = time.time()
    return generators[key]
def add_status(G: Generator):
    global generators, G_time
    ret = str(uuid4())
    generators[ret] = G
    G_time[ret] = time.time()
    if(len(generators)>200):
        st = sorted(list(generators), key=lambda x:G_time[x], reverse=True)[:200]
        _generators = {s: generators[s] for s in st}
        generators = _generators
    return ret

app = FastAPI()
INFER_LOCK = Lock()
class ContParam(BaseModel):
    feed: str = ""
    top_p: float = 0.4
    temperature: float = 1
    recall: list|NoneType = None
    stop_before: list|NoneType = None
    adjust: str|dict = ""
    length: int = 50
    min_length: int|NoneType = None
    stop_at_eot: bool = True
    ignore_occurrence: list|NoneType = None
_makedict = lambda **kwargs:kwargs
def _response(ret) -> JSONResponse:
    code = 200 if ret["status"] == 0 else -ret["status"]
    return JSONResponse(ret, status_code=code)

def make_response(**kwargs) -> JSONResponse:
    j = kwargs
    stat = j.get("status", 0)
    j["status"] = stat
    code = 200 if stat==0 else -stat
    return JSONResponse(j, status_code=code)


@app.post("/cont/{from_state}")
def post_continue(from_state: str, data: ContParam):
    global generators
    if(from_state=="new"):
        if(not data.feed):
            ret=_makedict(
                status=-1,
                message='Argument "feed" must be provided when creating new continuation Generator.'
            )
            return _response(ret)
        generator = init_generator(data.feed)
    else:
        if(from_state not in generators):
            ret=_makedict(
                status=-404,
                message="Generator '%s' not found."%(from_state)
            )
            return _response(ret)
        generator = get_status(from_state)
        if(data.feed):
            with INFER_LOCK:
                generator=generator.feed(data.feed)
    if(data.adjust):
        try:
            if(isinstance(data.adjust, str)):
                j = json.loads(data.adjust)
            elif(isinstance(data.adjust, dict)):
                j = data.adjust
            else:
                assert False, type(data.adjust)
            adj = {}
            for k, v in j.items():
                if(k=="<EOT>"):
                    adj[0] = adj.get(0, 0)+v
                else:
                    for t in tokenizer.encode(k):
                        adj[t] = adj.get(t, 0)+v
            generator = generator.derive(adjust=adj)
        except Exception as e:
            print(e)
    
    if(data.ignore_occurrence is not None):
        tokens = set()
        for idx, i in enumerate(data.ignore_occurrence):
            if(isinstance(i, str)):
                tokens.update(tokenizer.encode(i))
        ignore_occur_tokens = tokens
    else:
        ignore_occur_tokens = []

    if(data.recall):
        recall = [(0, tokenizer.encode(i)) for i in data.recall]
    else:
        recall = []
    if(data.stop_before):
        stop_before = [tokenizer.encode(i) for i in data.stop_before]
    else:
        stop_before = []
    if(data.min_length is not None):
        min_length = data.min_length
    else:
        min_length = 1
    G = generator.derive(
        top_p=data.top_p,
        temperature=data.temperature
    )
    init_G = G
    init_state = add_status(init_G)
    if(init_G.state is not None):
        init_G.debug_state(init_G.state)
    tokens = []
    Gs: List[Generator] = []
    def append(token, G):
        nonlocal tokens, generator
        tokens.append(token)
        Gs.append(G)
    def f_recall(forbid_tokens):
        nonlocal tokens, Gs
        le = len(forbid_tokens)
        assert len(tokens)>le
        assert tokens[-le:]==forbid_tokens
        tokens = tokens[:-le]
        Gs = Gs[:-le]
        G = Gs[-1]
        adj = {T:-abs(G.adjust.get(T, 0))-0.1 for T in forbid_tokens}
        return G.derive(adjust=adj)
    stopped = False
    with INFER_LOCK:
        for i in trange(data.length):
            if(stopped):
                break
            token, G = G.sample(ignore_occurence=ignore_occur_tokens)
            if(token==0):
                if(data.stop_at_eot):
                    if(len(tokens)>=min_length):
                        break
                # recall
                G = Gs[-1] if Gs else init_G        
                adj = {0:-0.1}
                G = G.derive(adjust=adj)
                continue
            append(token, G)
            contents = tokenizer.decode(tokens)
            for i in range(5):
                if(contents[-1] != "\ufffd"):
                    break
                token, G = G.sample()
                if(token==0):
                    adj = {0:-1}
                    G = Gs[-1] if Gs else init_G        
                    G = G.derive(adjust=adj)
                else:
                    append(token, G)
                    contents = tokenizer.decode(tokens)

            for idx, i in enumerate(recall):
                cnt, sb = i
                le = len(sb)
                if(len(tokens)>le and tokens[-le:]==sb):
                    print("recall", tokenizer.decode(sb), "from", contents)
                    G = f_recall(sb)
            for idx, i in enumerate(stop_before):
                sb = i
                le = len(sb)
                if(len(tokens)>le and tokens[-le:]==sb):
                    G = f_recall(sb)
                    stopped = len(tokens)>=min_length
                    if(stopped):
                        print("stop before" ,tokenizer.decode(sb), "from", contents, 'len_tokens', len(tokens))
                    else:
                        print("stop but not long enough" ,tokenizer.decode(sb), "from", contents)

    contents = tokenizer.decode(tokens)
    state = add_status(G)
    data = _makedict(
        init_state = init_state,
        state = state,
        contents = contents,
        full_history = tokenizer.decode(G.history)
        # tokens = tokens
    )
    ret = _makedict(
        status=0,
        data=data
    )
    return _response(ret)
        
class VecDistParam(BaseModel):
    query: str
    compare_with: list
    method: str = "centered_cosine"
    layers: list|NoneType = None
    states: list|NoneType = None

@app.get("/lora_strategy")
def get_lora_strategry(strategry: str):
    if(USING_LORA is None):
        return make_response(
            status=404,
            message="Not using LoRA"
        )
    else:
        fstrategy = get_fstrategy(strategry)
        USING_LORA.change_strategy(fstrategy)
        return make_response(
            status=0,
            message="OK"
        )

@app.post("/vec_dist")
def post_vec_dist(data: VecDistParam):
    query = data.query
    if(query not in generators):
        j = _makedict(
            status=-404,
            query=query,
            message="Generator %s not found"%query
        )
        resp = _response(j)
        return resp
    else:
        # query = generators[query]
        query = get_status(query)
    compare_with = {}
    for k in data.compare_with:
        if(k in generators):
            compare_with[k] = get_status(k)
        else:
            j = _makedict(
                status=-404,
                compare_with=k,
                message="Generator %s not found"%k
            )
            resp = _response(j)
            return resp
    
    layers = data.layers if data.layers is not None else [0.5]
    states = data.states if data.states is not None else [4]
    def get_state(G: Generator):
        nonlocal layers, states
        S = G.state
        _ = _makedict(
            sx=0,
            aa=1,
            bb=2,
            pp=3,
            ffn=4
        )
        ret = []
        for layer_idx in layers:
            if(0<layer_idx and layer_idx<1):
                layer_idx = int(layer_idx*len(states)/5)
            for state_idx in states:
                state_idx = _.get(state_idx, state_idx)
                ret.append(S[layer_idx*5+state_idx].to(torch.float32).cpu().numpy())
        return ret



    # if(data.method=="center_dot"):
    def f(g0, g1):
        ss0 = get_state(g0)
        ss1 = get_state(g1)
        ret = 0
        for idx, s0 in enumerate(ss0):
            s1: np.ndarray = ss1[idx]
            s0: np.ndarray = s0-s0.mean()
            s1: np.ndarray = s1-s1.mean()
            nm0 = (s0**2).sum()**0.5
            nm1 = (s1**2).sum()**0.5
            dist = (s0*s1).sum()/nm0/nm1
            dist = (1-dist)/2
            ret+=dist
        return ret/len(ss0)

    dists = {k: f(query, v) for k, v in compare_with.items()}

    ret = _makedict(
        status=0,
        data=dists
    )
    return _response(ret)


            
    
