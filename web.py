import gradio as gr
from modules.my_rwkv import Generator
from modules.initial import init_generator, tokenizer

states = {}

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    textbox = gr.Textbox()
    clr = gr.Button("Clear")
    rollback = gr.Button("撤回")
    def respond(message, chat_history, prog = gr.Progress()):
        key = tuple([(i.strip(" \n"), j.strip(" \n")) for i, j in chat_history])
        print("start", chat_history, key, states, key in states)
        if(key in states):
            G: Generator = states[key]
        else:
            G: Generator = init_generator("")
            G = G.derive(adjust={0: -10})
        
        G = G.feed(f"User: {message}\n菜菜: ")

        stop_ats = ["User:", "User："]
        stop_ats = [tokenizer.encode(i) for i in stop_ats]
        ignore_occur = set()
        for i in ["\n", "User:", "菜菜:"]:
            tokens = tokenizer.encode(i)
            ignore_occur.update(tokens)

        steps = list(range(100))
        progbar = prog.tqdm(steps)

        token = None
        tokens = []
        history = []
        contents = ""

        def append(token, G):
            nonlocal history, tokens, contents
            history.append((token, G))
            tokens.append(token)
            contents = tokenizer.decode(tokens)
        def recall(recall, least=1):
            nonlocal tokens, history, contents, token, G
            n = len(tokens)
            m = len(recall)
            if(n>=m+least and tokens[-m:]==recall):
                tokens = tokens[:-m]
                history = history[:-m]
                contents = tokenizer.decode(tokens)
                token, G = history[-1]
                return True
            else:
                return False
                

        for i in progbar:
            token, G = G.sample(ignore_occurence=ignore_occur)
            append(token, G)
            if(recall([0])):
               break
            for i in range(5):
                if(contents[-1]!='\ufffd'):
                    break
                token, G = G.sample(ignore_occurence=ignore_occur)
                append(token, G)
            progbar.desc = contents
            for stop_at in stop_ats:
                stopped = False
                if(recall(stop_at)):
                    stopped = True
                    break
            if(stopped):
                break
            
            
        chat_history.append((message, contents))
        print("end", chat_history)
        key = tuple([(i.strip(" \n"), j.strip(" \n")) for i, j in chat_history])
        states[key] = G
        print(tokenizer.decode(G.history))
        return "", chat_history

    textbox.submit(respond, [textbox, chatbot], [textbox, chatbot])
    clr.click(lambda :None, None, chatbot)
    rollback.click(lambda history: (history[:-1], history[-1][0]), chatbot, [chatbot, textbox])
if(__name__=="__main__"):
    demo.queue().launch(share=True)