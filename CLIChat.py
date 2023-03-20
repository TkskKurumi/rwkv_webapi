from modules.my_rwkv import tokenizer
from modules.initial import init_english_qa, init_neko, init_generator
from modules.initial import init_neko_a
def input_multiline(p="", cont=""):
    ret = [input(p)]
    while(ret[-1][-2:]=="\\n"):
        print(cont, end="")
        ret.append(input())
    return "".join(ret).replace("\\n", "\n")

if(__name__=="__main__"):
    QA = init_english_qa(user_name="User", bot_name="ChatRWKV")
    NEKO = init_neko_a()
    cont = NEKO
    
    while(True):        
        i = input_multiline(p="[you] >>", cont="      ..")
        
        if(i.startswith("/q")):
            prompt = "User: "+i[2:].strip()+"\nChatRWKV:"
            cont = QA.feed(prompt)
            tokens = []
            for token, cont in cont.stream(50):
                tokens.append(token)
                if('\ufffd' not in tokenizer.decode(tokens)):
                    print(tokenizer.decode(tokens), end="", flush=True)
                    tokens = []
            if(tokens):
                print(tokenizer.decode(tokens))
            print()
        elif(i.startswith("/exec")):
            cmd = i[i.find(" "):].strip(" ")
            exec(cmd)
        elif(i.startswith("/neko")):
            prompt = i[i.find(" "):].strip(" ")
            cont = NEKO.feed("你: "+prompt)
            tokens = []
            for st, cont in cont.stream(50):
                tokens.append(st)
                if('\ufffd' not in tokenizer.decode(tokens)):
                    print(tokenizer.decode(tokens), end="", flush=True)
                    tokens = []
            if(tokens):
                print(tokenizer.decode(tokens))
            print()
        elif(i.startswith("/+cont")):
            prompt = i[i.find(" "):].strip(" ")
            if(prompt):
                cont = cont.next(prompt)
            tokens = []
            for st, cont in cont.stream(50):
                tokens.append(st)
                if('\ufffd' not in tokenizer.decode(tokens)):
                    print(tokenizer.decode(tokens), end="", flush=True)
                    tokens = []
            if(tokens):
                print(tokenizer.decode(tokens))
            print()
        elif(i.startswith("/cont")):
            prompt = i[i.find(" "):].strip(" ")
            cont = init_generator(prompt)
            tokens = []
            for st, cont in cont.stream(50):
                tokens.append(st)
                if('\ufffd' not in tokenizer.decode(tokens)):
                    print(tokenizer.decode(tokens), end="", flush=True)
                    tokens = []
            if(tokens):
                print(tokenizer.decode(tokens))
            print()
        else:
            if(i):
                cont = cont.feed(i)
            tokens = []
            for st, cont in cont.stream(50):
                tokens.append(st)
                if('\ufffd' not in tokenizer.decode(tokens)):
                    print(tokenizer.decode(tokens), end="", flush=True)
                    tokens = []
            if(tokens):
                print(tokenizer.decode(tokens))
            print()