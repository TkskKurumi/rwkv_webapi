import gradio as gr
# from modules.my_rwkv import Generator
# from modules.initial import init_generator, tokenizer
from client import Client

states = {}

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    textbox = gr.Textbox()
    with gr.Row():
        # select_myname = gr.Dropdown(["User", "千千"], allow_custom_value=False)
        rollback = gr.Button("撤回")
        clr = gr.Button("Clear")
    def respond(message, chat_history, prog = gr.Progress()):
        myname = "User"
        key = tuple([(i.strip(" \n"), j.strip(" \n")) for i, j in chat_history])
        sb = [
                f"{myname}:", f"{myname}：", f"菜菜:", f"菜菜："
            ]
        if(key in states):
            cl = states[key]
        else:
            cl = Client(stop_before=sb)
        if(cl.history and (not cl.history.endswith("\n"))):
            feed_lf = "\n"
        else:
            feed_lf = ""
        resp, cl = cl.cont(f"{feed_lf}{myname}: {message}\n菜菜: ", ignore_occurrence=sb)
        if(resp.status_code!=200):
            resp.raise_for_status()
        
        contents = cl.last_gen
        print(cl.history)
        chat_history.append((message, contents))
        key = tuple([(i.strip(" \n"), j.strip(" \n")) for i, j in chat_history])
        states[key] = cl
        return "", chat_history

    textbox.submit(respond, [textbox, chatbot], [textbox, chatbot])
    clr.click(lambda :None, None, chatbot)
    rollback.click(lambda history: (history[:-1], history[-1][0]), chatbot, [chatbot, textbox])
if(__name__=="__main__"):
    demo.queue().launch(share=True)