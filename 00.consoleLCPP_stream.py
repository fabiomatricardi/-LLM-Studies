# Chat with an intelligent assistant in your terminal  model/stablelm-2-zephyr-1_6b-Q8_0.gguf
from openai import OpenAI
import sys
from time import sleep


print("\033[95;3;6m")
print("1. Waiting 10 seconds for the API to load...")
from llama_cpp import Llama
llm = Llama(
            model_path='models/dolphin-2.9.3-qwen2-0.5b.Q8_0.gguf',
            n_gpu_layers=0,
            temperature=0.1,
            top_p = 0.5,
            n_ctx=8192,
            max_tokens=600,
            repeat_penalty=1.7,
            stop=["<|im_end|>","Instruction:","### Instruction:","###<user>","</user>"],
            verbose=False,
            )
print("2. Model dolphin-2.9.3-qwen2-0.5b.Q8_0.gguf loaded with LlamaCPP...")
print("\033[0m")  #reset all

history = [
    {"role": "system", "content": "You are KS-AI, an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. Always reply in the language of the instructions."},
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Hi, I am KS-AI your intelligent assistant. How can I help you today?"}
]
print("\033[92;1m")
counter = 1
while True:
    if counter > 5:
        history = [
            {"role": "system", "content": "You are KS-AI, an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. Always reply in the language of the instructions."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hi, I am KS-AI your intelligent assistant. How can I help you today?"}
        ]        
    userinput = ""
    print("\033[1;30m")  #dark grey
    print("Enter your text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[91;1m")  #red
    lines = sys.stdin.readlines()
    for line in lines:
        userinput += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break
    history.append({"role": "user", "content": userinput})
    print("\033[92;1m")

    new_message = {"role": "assistant", "content": ""}
    
    full_response = ""
    fisrtround = 0
    for chunk in llm.create_chat_completion(
        messages=history,
        temperature=0.25,
        repeat_penalty= 1.6,
        stop=['<|im_end|>','</s>',"<end_of_turn>"],
        max_tokens=600,
        stream=True,):
        try:
            if chunk["choices"][0]["delta"]["content"]:
                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                full_response += chunk["choices"][0]["delta"]["content"]                              
        except:
            pass        
    new_message["content"] = full_response
    history.append(new_message)  
    counter += 1  