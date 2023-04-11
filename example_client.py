import requests
_makedict = lambda **kwargs:kwargs
data = _makedict(
    feed="""
以下是一段千千和菜菜的对话，
千千: 你的名字叫做菜菜，是一只可爱的猫娘，介绍一下自己吧。
菜菜:
""",
    top_p=0.2,
    length=100,
    stop_before=["千千:"]
)
url = "http://localhost:8000/cont/new"
r = requests.post(url, json=data)
print(r.json()["data"])

# continue from previous
state = r.json()["data"]["state"]
data = _makedict(
    feed = "",
    length=100
)
url = f"http://localhost:8000/cont/{state}"
r = requests.post(url, json=data)
print(r.json()["data"])