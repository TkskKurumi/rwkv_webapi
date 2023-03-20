from modules.initial import init_english_qa, init_generator, init_neko_a, tokenizer, Generator
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from tqdm import trange
import copy
from uuid import uuid4
from threading import Lock
base_generators = {
    "猫娘": init_neko_a(),
    "QA": init_english_qa()
}
generators = copy.copy(base_generators)
INFER_LOCK = Lock()

app = FastAPI()


get_kwa = lambda **kwargs:kwargs

def add_status(G: Generator):
    ret = str(uuid4())
    generators[ret] = G
    if(len(generators)>25):
        for i in generators:
            if(i not in base_generators):
                generators.pop(i)
                break
    return ret

@app.get("/cont")
def get_cont(from_state: str="", contents: str="", n: int=100, top_p=0.2):
    if(from_state != ""):
        if(from_state in generators):
            g = generators.get(from_state)
            if(contents):
                g = g.feed(contents)
        else:
            ret = get_kwa(
                status = -404,
                message = "No such state."
            )
            return JSONResponse(ret)
    else:
        g = init_generator(contents)
    tokens = []
    g = g.derive(top_p=top_p)
    for t in trange(n):
        token, g = g.sample()
        tokens.append(token)

    state = add_status(g)

    data = get_kwa(
        tokens = tokens,
        content = tokenizer.decode(tokens),
        state = state
    )
    ret = {"status": 0, "data": data}
    return JSONResponse(ret)