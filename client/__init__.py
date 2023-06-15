import requests
HOST = "http://localhost:8001"
makeargs = lambda **kwargs: kwargs
class Client:
    def __init__(self, state="new", temperature=1.5, top_p=0.5, stop_before=None, recall=None, history="", last_gen="", last_feed=""):
        self.temperature = temperature
        self.top_p = top_p
        self.state = state
        self.stop_before = stop_before if stop_before is not None else []
        self.recall = recall if recall is not None else []
        self.history = history
        self.last_gen = last_gen
        self.last_feed = last_feed
    def derive(self, **kwargs):
        d = {}
        d.update(self.__dict__)
        for k, v in kwargs.items():
            d[k] = v
        return Client(**d)
    def cont(self, feed="", length=100, ignore_occurrence=None):
        if(ignore_occurrence is None):
            ignore_occurrence = []
        args = makeargs(
            feed=feed,
            top_p = self.top_p,
            temperature = self.temperature,
            stop_before = self.stop_before,
            recall = self.recall,
            length = length,
            ignore_occurrence=ignore_occurrence
        )
        url = "/".join([HOST, "cont", self.state])
        resp = requests.post(url, json=args)
        if(resp.status_code==200):
            j = resp.json()
            data = j["data"]
            return resp, self.derive(state=data["state"], last_feed=feed, last_gen=data["contents"], history=data["full_history"])
        else:
            return resp, self
    def dist(self, *args, layers=[-1, -2, -3], states=["aa", "bb", "pp"]):
        req_args = makeargs(
            query=self.state,
            compare_with=[arg.state for arg in args],
            layers=layers,
            states=states
        )
        url = "/".join([HOST, "vec_dist"])
        resp = requests.post(url, json=req_args)
        return resp, self
    
