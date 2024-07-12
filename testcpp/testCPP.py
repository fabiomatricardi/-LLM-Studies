from rich.markdown import Markdown
import warnings
warnings.filterwarnings(action='ignore')
from rich.console import Console
console = Console(width=100)


from llama_cpp import Llama
modelfile = 'h2o-danube2-1.8b-chat-Q5_K_M.gguf'
modelpath = 'models/'+modelfile
modelname  = 'h2o-danube2-1.8b-chat'
model_nctx = 8192
train_nctx = 8192
stops = ['</s>']
chat_template = True
repo = 'https://huggingface.co/h2oai/h2o-danube2-1.8b-chat-GGUF'
original_repo = 'https://huggingface.co/h2oai/h2o-danube2-1.8b-chat-GGUF'
basemodel = 'llama'
numOfParams = '1.8b'
prompt_format = """
<|prompt|>Why is drinking water so healthy?</s><|answer|>
"""
llm = Llama(
            model_path=modelpath,
            n_gpu_layers=0,
            temperature=0.1,
            top_p = 0.5,
            n_ctx=model_nctx,
            max_tokens=600,
            repeat_penalty=1.44,
            stop=stops,
            verbose=True,
            )
console.print(f'loading {modelname} with LlamaCPP...\n\n\n\n\n')
console.rule(f"[bold turquoise2]{modelname}")
#console.print(Markdown(f'## Model Name: {modelname}'))
console.print('[italic]Model File[/italic]: ' + f'[light_steel_blue1 bold]{modelfile}')
console.rule("[bold red2]Details")
console.print(f'[bold green1]Training Context Window: {train_nctx}')
console.print(f'Stop Tokens: [red1]{stops}')
console.print(f'This model has Chat Template? {chat_template}')
console.rule()
console.print(Markdown(f'### PROMPT FORMAT:'))
promptmarkdown = f"""
```
{prompt_format}
```
"""
console.print(Markdown(promptmarkdown))
console.rule()
console.print(Markdown(f'### Additional info:'))
promptmarkdown = f"""
```
GGUF Repo: {repo}
Original Repo: {original_repo}
Model architecture: {basemodel}
Number of parameters: {numOfParams}
```

"""
console.print(Markdown(promptmarkdown))
console.rule(characters='=')
console.print('\n\n\n\n')

a = input('press a key to exit...')
del llm

#########################################################################

'''
from llama_cpp import Llama
modelfile = 'stablelm-2-zephyr-1_6b-Q5_K_M.gguf'
modelpath = 'models/'+modelfile
modelname  = 'stablelm-2-zephyr-1_6b'
model_nctx = 4096
train_nctx = 4096
stops = ['<|endoftext|>']
chat_template = True
repo = 'https://huggingface.co/second-state/stablelm-2-zephyr-1.6b-GGUF/tree/main'
original_repo = 'https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b'
basemodel = 'stablelm'
numOfParams = '1.6b'
prompt_format = """
<|user|>
{prompt}<|endoftext|>
<|assistant|>
"""
llm = Llama(
            model_path=modelpath,
            n_gpu_layers=0,
            temperature=0.1,
            top_p = 0.5,
            n_ctx=model_nctx,
            max_tokens=600,
            repeat_penalty=1.44,
            stop=stops,
            verbose=True,
            )
console.print(f'loading {modelname} with LlamaCPP...\n\n\n\n\n')
console.rule(f"[bold turquoise2]{modelname}")
#console.print(Markdown(f'## Model Name: {modelname}'))
console.print('[italic]Model File[/italic]: ' + f'[light_steel_blue1 bold]{modelfile}')
console.rule("[bold red2]Details")
console.print(f'[bold green1]Training Context Window: {train_nctx}')
console.print(f'Stop Tokens: [red1]{stops}')
console.print(f'This model has Chat Template? {chat_template}')
console.rule()
console.print(Markdown(f'### PROMPT FORMAT:'))
promptmarkdown = f"""
```
{prompt_format}
```
"""
console.print(Markdown(promptmarkdown))
console.rule()
console.print(Markdown(f'### Additional info:'))
promptmarkdown = f"""
```
GGUF Repo: {repo}
Original Repo: {original_repo}
Model architecture: {basemodel}
Number of parameters: {numOfParams}
```

"""
console.print(Markdown(promptmarkdown))
console.rule(characters='=')
console.print('\n\n\n\n')

a = input('press a key to exit...')
del llm
'''

############################################################################
'''
from llama_cpp import Llama
modelfile = 'stablelm-2-zephyr-1_6b-Q5_K_M.gguf'
modelpath = 'models/'+modelfile
modelname  = 'stablelm-2-zephyr-1_6b'
model_nctx = 4096
train_nctx = 4096
stops = ['<|endoftext|>']
chat_template = True
repo = 'https://huggingface.co/second-state/stablelm-2-zephyr-1.6b-GGUF/tree/main'
original_repo = 'https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b'
basemodel = 'stablelm'
numOfParams = '1.6b'
prompt_format = """
<|user|>
{prompt}<|endoftext|>
<|assistant|>
"""
llm = Llama(
            model_path=modelpath,
            n_gpu_layers=0,
            temperature=0.1,
            top_p = 0.5,
            n_ctx=model_nctx,
            max_tokens=600,
            repeat_penalty=1.44,
            stop=stops,
            verbose=True,
            )
console.print(f'loading {modelname} with LlamaCPP...\n\n\n\n\n')
console.rule(f"[bold turquoise2]{modelname}")
#console.print(Markdown(f'## Model Name: {modelname}'))
console.print('[italic]Model File[/italic]: ' + f'[light_steel_blue1 bold]{modelfile}')
console.rule("[bold red2]Details")
console.print(f'[bold green1]Training Context Window: {train_nctx}')
console.print(f'Stop Tokens: [red1]{stops}')
console.print(f'This model has Chat Template? {chat_template}')
console.rule()
console.print(Markdown(f'### PROMPT FORMAT:'))
promptmarkdown = f"""
```
{prompt_format}
```
"""
console.print(Markdown(promptmarkdown))
console.rule()
console.print(Markdown(f'### Additional info:'))
promptmarkdown = f"""
```
GGUF Repo: {repo}
Original Repo: {original_repo}
Model architecture: {basemodel}
Number of parameters: {numOfParams}
```

"""
console.print(Markdown(promptmarkdown))
console.rule(characters='=')
console.print('\n\n\n\n')

a = input('press a key to exit...')
del llm

'''
'''
from llama_cpp import Llama
modelpath = 'models/Aira-2-355M.Q6_K.gguf'
modelfile = 'Aira-2-355M.Q6_K.gguf'
modelname  = 'Aira-2-355M'
model_nctx = 1024
train_nctx = 1024
stops = ['<|endoftext|>']
chat_template = False
repo = 'https://huggingface.co/Felladrin/gguf-Aira-2-355M/tree/main'
original_repo = 'https://huggingface.co/nicholasKluge/Aira-2-355M'
basemodel = 'gpt2'
numOfParams = '355M'
prompt_format = """
<|startofinstruction|>What is a language model?<|endofinstruction|>
A language model is a probability distribution over a vocabulary.<|endofcompletion|>


Recommended inference parameters
penalty_alpha: 0.5
top_k: 2
repetition_penalty: 1.0016
"""
llm = Llama(
            model_path=modelpath,
            n_gpu_layers=0,
            temperature=0.1,
            top_p = 0.5,
            n_ctx=model_nctx,
            max_tokens=600,
            repeat_penalty=1.44,
            stop=stops,
            verbose=True,
            )
console.print(f'loading {modelname} with LlamaCPP...\n\n\n\n\n')
console.rule(f"[bold turquoise2]{modelname}")
#console.print(Markdown(f'## Model Name: {modelname}'))
console.print('[italic]Model File[/italic]: ' + f'[light_steel_blue1 bold]{modelfile}')
console.rule("[bold red2]Details")
console.print(f'[bold green1]Training Context Window: {train_nctx}')
console.print(f'Stop Tokens: [red1]{stops}')
console.print(f'This model has Chat Template? {chat_template}')
console.rule()
console.print(Markdown(f'### PROMPT FORMAT:'))
promptmarkdown = f"""
```
{prompt_format}
```
"""
console.print(Markdown(promptmarkdown))
console.rule()
console.print(Markdown(f'### Additional info:'))
promptmarkdown = f"""
```
GGUF Repo: {repo}
Original Repo: {original_repo}
Model architecture: {basemodel}
Number of parameters: {numOfParams}
```

"""
console.print(Markdown(promptmarkdown))
console.rule(characters='=')
console.print('\n\n\n\n')

a = input('press a key to exit...')
del llm
'''


#####################################################################################
'''
from llama_cpp import Llama
modelpath = 'models/smol-llama-101m-chat-v1.q8_0.gguf'
modelfile = 'smol-llama-101m-chat-v1.q8_0.gguf'
modelname  = 'smol-llama-101m-chat-v1'
model_nctx = 1024
train_nctx = 1024
stops = ['</s>']
chat_template = True
repo = 'https://huggingface.co/afrideva/Smol-Llama-101M-Chat-v1-GGUF'
original_repo = 'https://huggingface.co/Felladrin/Smol-Llama-101M-Chat-v1'
basemodel = 'llama'
numOfParams = '101M'
prompt_format = """
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant

Recommended inference parameters
penalty_alpha: 0.5
top_k: 4
repetition_penalty: 1.105
"""
llm = Llama(
            model_path=modelpath,
            n_gpu_layers=0,
            temperature=0.1,
            top_p = 0.5,
            n_ctx=model_nctx,
            max_tokens=350,
            repeat_penalty=1.44,
            stop=stops,
            verbose=True,
            )
console.print(f'loading {modelname} with LlamaCPP...\n\n\n\n\n')
console.rule(f"[bold turquoise2]{modelname}")
#console.print(Markdown(f'## Model Name: {modelname}'))
console.print('[italic]Model File[/italic]: ' + f'[light_steel_blue1 bold]{modelfile}')
console.rule("[bold red2]Details")
console.print(f'[bold green1]Training Context Window: {train_nctx}')
console.print(f'Stop Tokens: [red1]{stops}')
console.print(f'This model has Chat Template? {chat_template}')
console.rule()
console.print(Markdown(f'### PROMPT FORMAT:'))
promptmarkdown = f"""
```
{prompt_format}
```
"""
console.print(Markdown(promptmarkdown))
console.rule()
console.print(Markdown(f'### Additional info:'))
promptmarkdown = f"""
```
GGUF Repo: {repo}
Original Repo: {original_repo}
Model architecture: {basemodel}
Number of parameters: {numOfParams}
```

"""
console.print(Markdown(promptmarkdown))
console.rule(characters='=')
console.print('\n\n\n\n')

a = input('press a key to exit...')
del llm

'''
####################################################################################
'''
from llama_cpp import Llama
modelpath = 'models/Pythia-31M-Chat-v1.Q8_0.gguf'
modelfile = 'Pythia-31M-Chat-v1.Q8_0.gguf'
modelname  = 'Pythia-31M-Chat-v1'
model_nctx = 1024
train_nctx = 2048
stops = ['<|endoftext|>']
chat_template = True
repo = 'https://huggingface.co/Felladrin/gguf-Pythia-31M-Chat-v1/tree/main'
original_repo = 'https://huggingface.co/Felladrin/Pythia-31M-Chat-v1'
basemodel = 'gptneox'
numOfParams = '31M'
prompt_format = """
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant


Recommended inference parameters
penalty_alpha: 0.5
top_k: 2
repetition_penalty: 1.0016
"""
llm = Llama(
            model_path=modelpath,
            n_gpu_layers=0,
            temperature=0.1,
            top_p = 0.5,
            n_ctx=model_nctx,
            max_tokens=600,
            repeat_penalty=1.44,
            stop=stops,
            verbose=True,
            )
console.print(f'loading {modelname} with LlamaCPP...\n\n\n\n\n')
console.rule(f"[bold turquoise2]{modelname}")
#console.print(Markdown(f'## Model Name: {modelname}'))
console.print('[italic]Model File[/italic]: ' + f'[light_steel_blue1 bold]{modelfile}')
console.rule("[bold red2]Details")
console.print(f'[bold green1]Training Context Window: {train_nctx}')
console.print(f'Stop Tokens: [red1]{stops}')
console.print(f'This model has Chat Template? {chat_template}')
console.rule()
console.print(Markdown(f'### PROMPT FORMAT:'))
promptmarkdown = f"""
```
{prompt_format}
```
"""
console.print(Markdown(promptmarkdown))
console.rule()
console.print(Markdown(f'### Additional info:'))
promptmarkdown = f"""
```
GGUF Repo: {repo}
Original Repo: {original_repo}
Model architecture: {basemodel}
Number of parameters: {numOfParams}
```

"""
console.print(Markdown(promptmarkdown))
console.rule(characters='=')
console.print('\n\n\n\n')

a = input('press a key to exit...')
del llm

######################################################################################
from llama_cpp import Llama
modelpath = 'models/Aira-2-124M-DPO.Q8_0.gguf'
modelfile = 'Aira-2-124M-DPO.Q8_0.gguf'
modelname  = 'Aira-2-124M-DPO'
model_nctx = 1024
train_nctx = 1024
stops = ['<|endofcompletion|>']
chat_template = False
repo = 'https://huggingface.co/Felladrin/gguf-Aira-2-124M-DPO'
original_repo = 'https://huggingface.co/nicholasKluge/Aira-2-124M-DPO'
prompt_format = """
<|startofinstruction|>What is a language model?<|endofinstruction|>
A language model is a probability distribution over a vocabulary.<|endofcompletion|>
"""
basemodel = 'gpt2'
numOfParams = '124M'

llm = Llama(
            model_path=modelpath,
            n_gpu_layers=0,
            temperature=0.1,
            top_p = 0.5,
            n_ctx=model_nctx,
            max_tokens=600,
            repeat_penalty=1.44,
            stop=stops,
            verbose=True,
            )
console.print(f'loading {modelname} with LlamaCPP...\n\n\n\n\n')
console.rule(f"[bold turquoise2]{modelname}")
#console.print(Markdown(f'## Model Name: {modelname}'))
console.print('[italic]Model File[/italic]: ' + f'[light_steel_blue1 bold]{modelfile}')
console.rule("[bold red2]Details")
console.print(f'[bold green1]Training Context Window: {train_nctx}')
console.print(f'Stop Tokens: [red1]{stops}')
console.print(f'This model has Chat Template? {chat_template}')
console.rule()
console.print(Markdown(f'### PROMPT FORMAT:'))
promptmarkdown = f"""
```
{prompt_format}
```
"""
console.print(Markdown(promptmarkdown))
console.rule()
console.print(Markdown(f'### Additional info:'))
promptmarkdown = f"""
```
GGUF Repo: {repo}
Original Repo: {original_repo}
Model architecture: {basemodel}
Number of parameters: {numOfParams}
```

"""
console.print(Markdown(promptmarkdown))
console.rule(characters='=')
console.print('\n\n\n\n')

a = input('press a key to exit...')
del llm
######################################################################################



'''