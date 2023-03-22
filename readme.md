# rwkv_api
## Generator class
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

## main.py
fast-api web api.
