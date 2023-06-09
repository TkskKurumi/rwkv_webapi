from . import Client
if(__name__=="__main__"):
    cl = Client()
    results = []
    for t in [
        "千千：给你一拳", "千千：打你", "千千: 揍你",
        "千千：你好", "千千: 上午好", "千千：早上好",
        "千千：你好啊老婆", "千千：亲爱的", "千千：老婆"
        ]:
        data, new_cl = cl.cont(t, length=0)
        results.append(new_cl)
    for new_cl in list(results):
        results.append(new_cl.cont(length=25)[-1])
    for cur_cl in results:
        cmp_with = [i for i in results if i is not cur_cl]
        data, cur_cl = cur_cl.dist(*cmp_with, layers=[-1, -2, -3], states=["aa", "bb", "pp"])
        dists = [(dist, cmp_with[idx].history) for idx, dist in enumerate(data)]
        print(repr(cur_cl.history), "->", repr(min(dists)[-1]))