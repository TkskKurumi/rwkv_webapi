from modules.initial import init_english_qa, init_generator, init_neko_a, tokenizer, Generator
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from tqdm import trange
import copy, json
from typing import List
from uuid import uuid4
from threading import Lock
from types import NoneType
from pydantic import BaseModel

sanity_check = init_generator("Sanity Check")

base_generators = {
}
generators = {
    **base_generators
}


def add_status(G: Generator):
    ret = str(uuid4())
    generators[ret] = G
    if(len(generators)>100):
        for i in generators:
            generators.pop(i)
            break
        for k, v in base_generators.items():
            if(k not in generators):
                generators[k] = v
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
        generator = generators[from_state]
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
            token, G = G.sample(ignore_occurence=data.ignore_occurrence)
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
            if(contents[-1] == "\ufffd"):
                token, G = G.sample()
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
        
    