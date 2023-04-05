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
base_generators = {
    "猫娘": init_neko_a(),
    "QA": init_english_qa()
}
generators = {
    **base_generators
}


def add_status(G: Generator):
    ret = str(uuid4())
    generators[ret] = G
    if(len(generators)>25):
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
    temperature: float = 2
    recall: list|NoneType = None
    adjust: str = ""
    length: int = 50
    stop_at_eot: bool = True

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
            generator=generator.feed(data.feed)
    
    if(data.recall):
        recall = [(0, tokenizer.encode(i)) for i in data.recall]
    else:
        recall = []

    G = generator.derive(
        top_p=data.top_p,
        temperature=data.temperature
    )
    init_G = G
    init_state = add_status(init_G)
    tokens = []
    Gs: List[Generator] = []
    def append(token, G):
        nonlocal tokens, generator
        tokens.append(token)
        Gs.append(G)
    for i in trange(data.length):
        token, G = G.sample()
        if(token==0 and data.stop_at_eot):
            break
        append(token, G)
        contents = tokenizer.decode(tokens)
        if(contents[-1] == "\ufffd"):
            token, G = G.sample()
            append(token, G)
            contents = tokenizer.decode(tokens)

        if(recall):
            for idx, i in enumerate(recall):
                cnt, sb = i
                le = len(sb)
                stopped = False
                if(len(tokens)>le and tokens[-le:]==sb):
                    print("recall", tokenizer.decode(sb), "from", contents)
                    recall[idx] = (cnt+1, sb)
                    tokens = tokens[:-le]
                    G = Gs[-le-1]
                    if(cnt>=2):
                        stopped = True
                        break    
                    adj = {T:-0.1 for T in sb}
                    G = G.derive(adjust=adj)
                if(stopped):
                    break

    contents = tokenizer.decode(tokens)
    state = add_status(G)
    data = _makedict(
        init_state = init_state,
        state = state,
        contents = contents
        # tokens = tokens
    )
    ret = _makedict(
        status=0,
        data=data
    )
    return _response(ret)
        
    