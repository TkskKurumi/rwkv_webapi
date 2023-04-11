# rwkv_api
## main.py
fast-api web api
```shell
uvicorn main:app
```

## client example
### example_client.py
Example for web api client.

### example QQ bot
[yaqianbot](https://github.com/TkskKurumi/yaqianbot_v2/blob/dev/yaqianbot/plugins/plg_rwkvapi.py)


## Generator Class
For ease of use, introduce the Generator for status control.

It contains following datas:
+ RNN state (context).
+ RNN previous output logits.
+ sampling parameters.
+ + top_p
  + logits adjustment
  + frequency record & frequency penalty

It provides following methods:
+ feed, feed prompts into it and returns new Generator with status after feed
+ stream, predict outputs

## CLIChat.py
CLI chat test.
