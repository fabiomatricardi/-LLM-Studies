import streamlit as st
import datetime
import os
from io import StringIO
from rich.markdown import Markdown
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
console = Console(width=90)
import tiktoken
import random
import string

encoding = tiktoken.get_encoding("r50k_base") #context_count = len(encoding.encode(yourtext))

from llama_cpp import Llama

#AVATARS  👷🐦  🥶🌀
av_us = '🧑‍💻'  #"🦖"  #A single emoji, e.g. "🧑‍💻", "🤖", "🦖". Shortcodes are not supported.
av_ass = '🤖'

# Set the webpage title
st.set_page_config(
    page_title="Your LocalGPT with 🟠 Qwen-0.5",
    page_icon="🟠",
    layout="wide")

@st.cache_resource 
def create_chat():   
# Set HF API token  and HF repo
    from llama_cpp import Llama
    qwen05b = Llama(
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
    print('loading qwen2-0_5b-instruct-q8_0.gguf with LlamaCPP...')
    return qwen05b

def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()



# Create a header element
mytitle = '<p style="color:Yellow; font-size: 32px;text-align:center;"><b>Your own LocalGPT with 🟠 Key Solution AI</b></p>'
st.markdown(mytitle, unsafe_allow_html=True)
#st.header("Your own LocalGPT with 🌀 h2o-danube-1.8b-chat")
subtitle = '<p style="color:DeepSkyBlue; font-size: 28px;text-align:center;"><b><i>Powerwed by Qwen, the best 0.5B chat model?</i></b></p>'
st.markdown(subtitle, unsafe_allow_html=True)


def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res

# create THE SESSIoN STATES
if "logfilename" not in st.session_state:
## Logger file
    logfile = f'{genRANstring(5)}_log.txt'
    st.session_state.logfilename = logfile
    #Write in the history the first 2 sessions
    writehistory(st.session_state.logfilename,f'{str(datetime.datetime.now())}\n\nYour own LocalGPT with 🌀 Qwen-0.5b-chat\n---\n🧠🫡: You are a helpful assistant.')    
    writehistory(st.session_state.logfilename,f'🌀: How may I help you today?')

if "len_context" not in st.session_state:
    st.session_state.len_context = 0

if "limiter" not in st.session_state:
    st.session_state.limiter = 0

if "bufstatus" not in st.session_state:
    st.session_state.bufstatus = "**:green[Good]**"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.1

if "maxlength" not in st.session_state:
    st.session_state.maxlength = 500

# Point to the local server
llm = create_chat()
 
# CREATE THE SIDEBAR
with st.sidebar:
    st.image('logo.png', use_column_width=True)
    st.session_state.temperature = st.slider('Temperature:', min_value=0.0, max_value=1.0, value=0.1, step=0.02)
    st.session_state.limiter = st.slider('Turns:', min_value=7, max_value=17, value=12, step=1)
    st.session_state.maxlength = st.slider('Length reply:', min_value=150, max_value=1000, 
                                           value=500, step=50)
    mytokens = st.markdown(f"""**Context turns** {st.session_state.len_context}""")
    st.markdown(f"Buffer status: {st.session_state.bufstatus}")
    st.markdown(f"**Logfile**: {st.session_state.logfilename}")
    btnClear = st.button("Clear History",type="primary", use_container_width=True)

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are QWEN-KS, a helpful assistant. You reply only to the user questions. You always reply in the language of the instructions.",},
        {"role": "user", "content": "Hi, I am a Key Solution employee, a Company that works in the Oil and Gas sector."},
        {"role": "assistant", "content": "Hi there, I am QWEN-KS, how may I help you today?"}
    ]

def clearHistory():
    st.session_state.messages = [
        {"role": "system", "content": "You are QWEN-KS, a helpful assistant. You reply only to the user questions. You always reply in the language of the instructions.",},
        {"role": "user", "content": "Hi, I am a Key Solution employee, a Company that works in the Oil and Gas sector."},
        {"role": "assistant", "content": "Hi there, I am QWEN-KS, how may I help you today?"}
    ]
    st.session_state.len_context = len(st.session_state.messages)
if btnClear:
      clearHistory()  
      st.session_state.len_context = len(st.session_state.messages)

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages[1:]:
    if message["role"] == "user":
        with st.chat_message(message["role"],avatar=av_us):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"],avatar=av_ass):
            st.markdown(message["content"])

# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here. Shift+Enter to add a new line", key="user_input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add our input to the chat window
    with st.chat_message("user", avatar=av_us):
        st.markdown(user_prompt)
        writehistory(st.session_state.logfilename,f'👷: {user_prompt}')

    
    with st.chat_message("assistant",avatar=av_ass):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            response = ''
            conv_messages = []
            st.session_state.len_context = len(st.session_state.messages) 
            # Checking context window for the LLM, not for the chat history to be displayed
            if st.session_state.len_context > st.session_state.limiter:
                st.session_state.bufstatus = "**:red[Overflow]**"
                # this will keep 5 full turns into consideration 
                x=st.session_state.limiter-5
                conv_messages.append(st.session_state.messages[0])
                for i in range(0,x):
                    conv_messages.append(st.session_state.messages[-x+i])
                print(len(conv_messages))
                full_response = ""
                for chunk in llm.create_chat_completion(
                    messages=conv_messages,
                    temperature=st.session_state.temperature,
                    repeat_penalty= 1.6,
                    stop=['<|im_end|>','</s>',"<end_of_turn>"],
                    max_tokens=st.session_state.maxlength,
                    stream=True,):
                    try:
                        if chunk["choices"][0]["delta"]["content"]:
                            full_response += chunk["choices"][0]["delta"]["content"]
                            message_placeholder.markdown(full_response + "🟠")                                 
                    except:
                        pass       
                message_placeholder.markdown(full_response)
                writehistory(st.session_state.logfilename,f'🟠: {full_response}\n\n---\n\n') 
            else:
                st.session_state.bufstatus = "**:green[Good]**"
                full_response = ""
                for chunk in llm.create_chat_completion(
                    messages=st.session_state.messages,
                    temperature=st.session_state.temperature,
                    repeat_penalty= 1.6,
                    stop=['<|im_end|>','</s>',"<end_of_turn>"],
                    max_tokens=st.session_state.maxlength,
                    stream=True,):
                    try:
                        if chunk["choices"][0]["delta"]["content"]:
                            full_response += chunk["choices"][0]["delta"]["content"]
                            message_placeholder.markdown(full_response + "🟠")                                 
                    except:
                        pass                 
                message_placeholder.markdown(full_response)
                writehistory(st.session_state.logfilename,f'🟠: {full_response}\n\n---\n\n') 
            
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
    st.session_state.len_context = len(st.session_state.messages)
