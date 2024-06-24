# Chat with an intelligent assistant in your terminal  model/stablelm-2-zephyr-1_6b-Q8_0.gguf
from openai import OpenAI
import sys
from time import sleep
from llama_cpp import Llama
import datetime

def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

timestamp = datetime.datetime.now()
log = f"""




#########################################################################
{str(timestamp)}

TEST THE SAME PROMPT ON MULTIPLE GGUF 
SMALL LANGUAGE MODELS

---
Outline the content for an Article about the main 3 problems the AI research is facing. The title is 'Tehe AI 3-body problem. On Benchmarks, Computation and Data: if AGI progress has stalled, what is the future ahead?' 
The aritcle must cover the following topics:
1) Benchmarks are like a snake that bites its own tail - AI is becoming the judge for other new LLM performance review.
2) skills are not intelligence
3) computation resources required to run large Language Models are expensive, and training requires long time. Don't we have new technologies to allow normal consumer hardware available for the task?
4) Data: every AI is trained on available data, and we have saturated the available one. How we can ensure truthfulness and quality? garbage is is garbage out

Format your output as a list.

Outline:


Write the content for an Article about the main 3 problems the AI research is facing. 
The title is 'The AI 3-body problem. On Benchmarks, Computation and Data: if AGI progress has stalled, what is the future ahead?' 
The article must be 2000 words and must include the following topics:
- Benchmarks are like a snake that bites its own tail and AI is becoming the judge for other new LLM performance review.
- skills are not intelligence
- computation resources required to run large Language Models are expensive, and training requires long time. Don't we have new technologies to allow normal consumer hardware available for the task?
- Data: every AI is trained on available data, and we have saturated the available one. How we can ensure truthfulness and quality? garbage is is garbage out


Article:

---

#########################################################################

"""
writehistory('ModelsTest_history.txt',log)



history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, How can I help you today?"}
]
print("\033[92;1m")


userinput = """Write the content for an Article about the main 3 problems the AI research is facing. 
The title is 'The AI 3-body problem. On Benchmarks, Computation and Data: if AGI progress has stalled, what is the future ahead?' 
The article must be 2000 words and must include the following topics:
- Benchmarks are like a snake that bites its own tail and AI is becoming the judge for other new LLM performance review.
- skills are not intelligence
- computation resources required to run large Language Models are expensive, and training requires long time. Don't we have new technologies to allow normal consumer hardware available for the task?
- Data: every AI is trained on available data, and we have saturated the available one. How we can ensure truthfulness and quality? garbage is is garbage out


Article:"""

# Start the test with 5 different models
###########################################



##################  dolphin-2.9.3-qwen2-0.5b.Q8_0.gguf ###########################
print("\033[95;3;6m")
print("1. Loading dolphin-2.9.3-qwen2-0.5b.Q8_0.gguf...")
llm = Llama(
            model_path='models/dolphin-2.9.3-qwen2-0.5b.Q8_0.gguf',
            n_gpu_layers=0,
            temperature=0.1,
            top_p = 0.5,
            n_ctx=8192,
            max_tokens=600,
            repeat_penalty=1.7,
            stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
            verbose=True,
            )
print("2. Model loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, How can I help you today?"}
]        
history.append({"role": "user", "content": userinput})
print("\033[92;1m")
start = datetime.datetime.now()    
full_response = ""
fisrtround = 0
print("GENERATED dolphin-2.9.3-qwen2-0.5b.Q8_0.gguf...\n-------------------------------------------\n")
for chunk in llm.create_chat_completion(
    messages=history,
    temperature=0.35,
    repeat_penalty= 1.6,
    stop=['<|im_end|>','</s>',"<end_of_turn>"],
    max_tokens=2000,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
delta = datetime.datetime.now() - start
log = f"""
========================================================================
dolphin-2.9.3-qwen2-0.5b.Q8_0.gguf
------------------------------------------------------------------------
generated in {str(delta)}
---
{full_response}
------------------------------------------------------------------------


"""
writehistory('ModelsTest_history.txt',log)
del llm

##################  models\h2o-danube2-1.8b-chat-Q8_0.gguf ###########################
print("\033[95;3;6m")
print("1. Loading h2o-danube2-1.8b-chat-Q8_0.gguf...")
llm = Llama(
            model_path='models/h2o-danube2-1.8b-chat-Q8_0.gguf',
            n_gpu_layers=0,
            temperature=0.35,
            top_p = 0.5,
            n_ctx=8192,
            max_tokens=2000,
            repeat_penalty=1.7,
            stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
            verbose=True,
            )
print("2. Model loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, How can I help you today?"}
]        
history.append({"role": "user", "content": userinput})
print("\033[92;1m")
start = datetime.datetime.now()    
full_response = ""
print("GENERATED h2o-danube2-1.8b-chat-Q8_0.gguf...\n-------------------------------------------\n")
fisrtround = 0
for chunk in llm.create_chat_completion(
    messages=history,
    temperature=0.35,
    repeat_penalty= 1.6,
    stop=['<|im_end|>','</s>',"<end_of_turn>"],
    max_tokens=2000,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
delta = datetime.datetime.now() - start
log = f"""
========================================================================
h2o-danube2-1.8b-chat-Q8_0.gguf
------------------------------------------------------------------------
generated in {str(delta)}
---
{full_response}
------------------------------------------------------------------------


"""
writehistory('ModelsTest_history.txt',log)
del llm

##################  models\PowerQwen-1.5B-v1.Q4_K_M.gguf ###########################
#skipped for template problems
##################  models\Phi-3-mini-128k-instruct-Q4_K_M.gguf ###########################
print("\033[95;3;6m")
print("1. Loading Phi-3-mini-128k-instruct-Q4_K_M.gguf...")
llm = Llama(
            model_path='models/Phi-3-mini-128k-instruct-Q4_K_M.gguf',
            n_gpu_layers=0,
            temperature=0.35,
            top_p = 0.5,
            n_ctx=8192,
            max_tokens=2000,
            repeat_penalty=1.7,
            stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
            verbose=True,
            )
print("2. Model loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, How can I help you today?"}
]        
history.append({"role": "user", "content": userinput})
print("\033[92;1m")
start = datetime.datetime.now()    
full_response = ""
print("GENERATED Phi-3-mini-128k-instruct-Q4_K_M.gguf...\n-------------------------------------------\n")
fisrtround = 0
for chunk in llm.create_chat_completion(
    messages=history,
    temperature=0.35,
    repeat_penalty= 1.6,
    stop=['<|im_end|>','</s>',"<end_of_turn>"],
    max_tokens=2000,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
delta = datetime.datetime.now() - start
log = f"""
========================================================================
Phi-3-mini-128k-instruct-Q4_K_M.gguf
------------------------------------------------------------------------
generated in {str(delta)}
---
{full_response}
------------------------------------------------------------------------


"""
writehistory('ModelsTest_history.txt',log)
del llm

##################  models\Qwen1.5-0.5B-Chat_llamafy.Q8_0.gguf ###########################
print("\033[95;3;6m")
print("1. Loading Qwen1.5-0.5B-Chat_llamafy.Q8_0.gguf...")
llm = Llama(
            model_path='models/Qwen1.5-0.5B-Chat_llamafy.Q8_0.gguf',
            n_gpu_layers=0,
            temperature=0.35,
            top_p = 0.5,
            n_ctx=8192,
            max_tokens=2000,
            repeat_penalty=1.7,
            stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
            verbose=True,
            )
print("2. Model loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, How can I help you today?"}
]        
history.append({"role": "user", "content": userinput})
print("\033[92;1m")
start = datetime.datetime.now()    
full_response = ""
print("GENERATED Qwen1.5-0.5B-Chat_llamafy.Q8_0.gguf...\n-------------------------------------------\n")
fisrtround = 0
for chunk in llm.create_chat_completion(
    messages=history,
    temperature=0.35,
    repeat_penalty= 1.6,
    stop=['<|im_end|>','</s>',"<end_of_turn>"],
    max_tokens=2000,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
delta = datetime.datetime.now() - start
log = f"""
========================================================================
Qwen1.5-0.5B-Chat_llamafy.Q8_0.gguf
------------------------------------------------------------------------
generated in {str(delta)}
---
{full_response}
------------------------------------------------------------------------


"""
writehistory('ModelsTest_history.txt',log)
del llm

##################  models\qwen2-deita-500m-q8_0.gguf ###########################
print("\033[95;3;6m")
print("1. Loading qwen2-deita-500m-q8_0.gguf...")
llm = Llama(
            model_path='models/qwen2-deita-500m-q8_0.gguf',
            n_gpu_layers=0,
            temperature=0.35,
            top_p = 0.5,
            n_ctx=8192,
            max_tokens=2000,
            repeat_penalty=1.7,
            stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
            verbose=True,
            )
print("2. Model loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = [
            {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, How can I help you today?"}
]        
history.append({"role": "user", "content": userinput})
print("\033[92;1m")
start = datetime.datetime.now()    
full_response = ""
print("GENERATED qwen2-deita-500m-q8_0.gguf...\n-------------------------------------------\n")
fisrtround = 0
for chunk in llm.create_chat_completion(
    messages=history,
    temperature=0.35,
    repeat_penalty= 1.6,
    stop=['<|im_end|>','</s>',"<end_of_turn>"],
    max_tokens=2000,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
delta = datetime.datetime.now() - start
log = f"""
========================================================================
qwen2-deita-500m-q8_0.gguf
------------------------------------------------------------------------
generated in {str(delta)}
---
{full_response}
------------------------------------------------------------------------


"""
writehistory('ModelsTest_history.txt',log)
del llm

##################  models\stablelm-2-zephyr-1_6b-Q5_K_M.gguf ###########################
print("\033[95;3;6m")
print("1. Loading stablelm-2-zephyr-1_6b-Q5_K_M.gguf...")
llm = Llama(
            model_path='models/stablelm-2-zephyr-1_6b-Q5_K_M.gguf',
            n_gpu_layers=0,
            temperature=0.35,
            top_p = 0.5,
            n_ctx=4096,
            max_tokens=2000,
            repeat_penalty=1.7,
            stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
            verbose=True,
            )
print("2. Model loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, How can I help you today?"}
]        
history.append({"role": "user", "content": userinput})
print("\033[92;1m")
start = datetime.datetime.now()    
full_response = ""
print("GENERATED stablelm-2-zephyr-1_6b-Q5_K_M.gguf...\n-------------------------------------------\n")
fisrtround = 0
for chunk in llm.create_chat_completion(
    messages=history,
    temperature=0.35,
    repeat_penalty= 1.6,
    stop=['<|im_end|>','</s>',"<end_of_turn>"],
    max_tokens=2000,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
delta = datetime.datetime.now() - start
log = f"""
========================================================================
stablelm-2-zephyr-1_6b-Q5_K_M.gguf
------------------------------------------------------------------------
generated in {str(delta)}
---
{full_response}
------------------------------------------------------------------------


"""
writehistory('ModelsTest_history.txt',log)
del llm


##################  models\TinyEnsemble-3x1.1B-TinyMoE.Q6_K.gguf ###########################
print("\033[95;3;6m")
print("1. Loading TinyEnsemble-3x1.1B-TinyMoE.Q6_K.gguf...")
llm = Llama(
            model_path='models/TinyEnsemble-3x1.1B-TinyMoE.Q6_K.gguf',
            n_gpu_layers=0,
            temperature=0.35,
            top_p = 0.5,
            n_ctx=4096,
            max_tokens=2000,
            repeat_penalty=1.7,
            stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
            verbose=True,
            )
print("2. Model loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, How can I help you today?"}
]        
history.append({"role": "user", "content": userinput})
print("\033[92;1m")
start = datetime.datetime.now()    
full_response = ""
print("GENERATED TinyEnsemble-3x1.1B-TinyMoE.Q6_K.gguf...\n-------------------------------------------\n")
fisrtround = 0
for chunk in llm.create_chat_completion(
    messages=history,
    temperature=0.35,
    repeat_penalty= 1.6,
    stop=['<|im_end|>','</s>',"<end_of_turn>"],
    max_tokens=2000,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
delta = datetime.datetime.now() - start
log = f"""
========================================================================
TinyEnsemble-3x1.1B-TinyMoE.Q6_K.gguf
------------------------------------------------------------------------
generated in {str(delta)}
---
{full_response}
------------------------------------------------------------------------


"""
writehistory('ModelsTest_history.txt',log)
del llm

##################  models\tinyllama-moe4-q8_0.gguf ###########################
print("\033[95;3;6m")
print("1. Loading tinyllama-moe4-q8_0.gguf...")
llm = Llama(
            model_path='models/tinyllama-moe4-q8_0.gguf',
            n_gpu_layers=0,
            temperature=0.35,
            top_p = 0.5,
            n_ctx=2048,
            max_tokens=1800,
            repeat_penalty=1.7,
            stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
            verbose=True,
            )
print("2. Model loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, How can I help you today?"}
]        
history.append({"role": "user", "content": userinput})
print("\033[92;1m")
start = datetime.datetime.now()    
full_response = ""
print("GENERATED tinyllama-moe4-q8_0.gguf...\n-------------------------------------------\n")
fisrtround = 0
for chunk in llm.create_chat_completion(
    messages=history,
    temperature=0.35,
    repeat_penalty= 1.6,
    stop=['<|im_end|>','</s>',"<end_of_turn>"],
    max_tokens=1800,
    stream=True,):
    try:
        if chunk["choices"][0]["delta"]["content"]:
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            full_response += chunk["choices"][0]["delta"]["content"]                              
    except:
        pass        
delta = datetime.datetime.now() - start
log = f"""
========================================================================
tinyllama-moe4-q8_0.gguf
------------------------------------------------------------------------
generated in {str(delta)}
---
{full_response}
------------------------------------------------------------------------


"""
writehistory('ModelsTest_history.txt',log)
del llm