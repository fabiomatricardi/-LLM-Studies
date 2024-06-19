# LLM-Studies
Collection of Resources, repositories and snippets for LLM and  Open source Generative AI



## LLM Frameworks and tools
- ### [Duckduckgo_search](https://pypi.org/project/duckduckgo-search/)
> Search for words, documents, images, videos, news, maps and text translation using the DuckDuckGo.com search engine. Downloading files and images to a local hard drive.


## OpenAI compatible API
<img src='https://python.langchain.com/v0.2/img/brand/wordmark.png' width=400>
- https://python.langchain.com/v0.2/docs/integrations/chat/openai/

```
# source https://github.com/fabiomatricardi/llamaCPP_Agents/blob/main/testapi.py
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1", api_key="not-needed",
    model="qwen1_5-0_5b-chat",
    temperature=0.1,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    # organization="...",
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
"""
messages = [
    {
        "role": "system",
        "content":"You are a helpful assistant that translates English to French. Translate the user sentence."},
    {"role":"human", "content":"I love programming."},
]
"""

ai_msg = llm.invoke(messages)
#ai_msg
print(ai_msg.content)
```

- https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/
- https://python.langchain.com/v0.1/docs/integrations/text_embedding/llamacpp/


## LlamaCPP
<img src='https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png' width=400>

Main project Repo on GitHub https://github.com/ggerganov/llama.cpp/tree/compilade/refactor-kv-cache?tab=readme-ov-file

#### Python bindings
https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file
> compatible openAI server https://llama-cpp-python.readthedocs.io/en/stable/server/

- [LLamaCPPChat](https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/)
- [LLamaCPPEmbeddings](https://python.langchain.com/v0.1/docs/integrations/text_embedding/llamacpp/)




## Agentic llm
- ### [LLMs: A New Era in Web Scraping](https://medium.com/@ireihani/llms-a-new-era-in-web-scraping-1ffc6b93abda)
<img src='https://miro.medium.com/v2/resize:fit:640/format:webp/1*SAvlcuWZnu_PIDbI5yKVyA.png' height=200>

> In the rapidly evolving landscape of machine learning and its real-world applications, a notable project has emerged on GitHub: GPT Researcher. This autonomous system functions as an advanced research assistant, drawing from the capabilities of AutoGPT and Plan-and-Solve to offer a unique approach to online research. One of its most promising applications is in the field of web scraping, a technique for extracting data from websites, where Large Language Models (LLMs) like GPT Researcher offer significant advantages.

https://docs.gptr.dev/docs/gpt-researcher/pip-package

- ### [CrewAI 🕷️🤖](https://github.com/unclecode/crawl4ai)
> Crawl4AI has one clear task: to simplify crawling and extract useful information from web pages, making it accessible for large language models (LLMs) and AI applications. 🆓🌐
> 
> [Colab Notebook](https://colab.research.google.com/drive/1wz8u30rvbq6Scodye9AGCw8Qg_Z8QGsk#scrollTo=GyRjyQ1UoaJr)

- ### [Python Scraper Github projects](https://github.com/topics/scraper?l=python)
> Here are 4,249 public repositories matching this topic...

- ### [AnimeDL](https://github.com/justfoolingaround/animdl)
<img src='https://camo.githubusercontent.com/3f4968553de25d951d8b57875296c4fe1e259a578c800a2e1283e589edd82f05/68747470733a2f2f696e7669646765742e737769746368626c6164652e78797a2f67615832536e7374326a' width=400>



## Miscellaneous
- ### [Python RICH](https://rich.readthedocs.io/en/latest/index.html)
> [Color codes](https://rich.readthedocs.io/en/latest/appendix/colors.html)

- ### [EasyGUI](https://easygui.readthedocs.io/en/master/index.html)
> EasyGUI is a module for very simple, very easy GUI programming in Python. EasyGUI is different from other GUI generators in that EasyGUI is NOT event-driven. Instead, all GUI interactions are invoked by simple function calls.<br>
> EasyGui provides an easy-to-use interface for simple GUI interaction with a user. It does not require the programmer to know anything about tkinter, frames, widgets, callbacks or lambda.<br>
> EasyGUI runs on Python 2 and 3, and does not have any dependencies.

<br><br><br>
<img src='https://scontent-mxp2-1.xx.fbcdn.net/v/t39.8562-6/252294889_575082167077436_6034106545912333281_n.svg/meta-logo-primary_standardsize.svg?_nc_cat=1&ccb=1-7&_nc_sid=e280be&_nc_ohc=gHe9SH5Q39oQ7kNvgF40WF2&_nc_ht=scontent-mxp2-1.xx&oh=00_AYC2pLd7XyxG2PbimvpfsgvSoenwfHnZ0smUp7kV8OLsew&oe=66788A39' width=200>
- ### [Meta Chameleon](https://github.com/facebookresearch/chameleon/tree/main)
> Annoucement here https://about.fb.com/news/2024/06/releasing-new-ai-research-models-to-accelerate-innovation-at-scale/ <br>and here https://twitter.com/AIatMeta/status/1803107817345393136 <br>
> As we shared in our research paper last month, Meta Chameleon is a family of models that can combine text and images as input and output any combination of text and images with a single unified architecture for both encoding and decoding. While most current late-fusion models use diffusion-based learning, Meta Chameleon uses tokenization for text and images. This enables a more unified approach and makes the model easier to design, maintain, and scale. The possibilities are endless—imagine generating creative captions for images or using a mix of text prompts and images to create an entirely new scene.<br.
> https://ai.meta.com/blog/meta-fair-research-new-releases/

- ### [ParlAI](https://parl.ai/projects/recipes/)
> https://github.com/facebookresearch/ParlAI<br>
>Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that scaling neural models in the number of parameters and the size of the data they are trained on gives improved results, we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to their partners, both asking and answering questions, and displaying knowledge, empathy and personality appropriately, depending on the situation. We show that large scale models can learn these skills when given appropriate training data and choice of generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter neural models, and make our models and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing failure cases of our models.

Text2Text encoder-decoder models:
> https://huggingface.co/facebook/blenderbot_small-90M/tree/main<br>
> https://huggingface.co/facebook/blenderbot-400M-distill/tree/main
```
https://parl.ai/projects/
https://parl.ai/projects/bb3x/

```


### Mixture of Experts or MoAgents
Quantized models GGUF [mradermacher/TinyEnsemble-3x1.1B-TinyMoE-GGUF](https://huggingface.co/mradermacher/TinyEnsemble-3x1.1B-TinyMoE-GGUF)


Original model [TinyEnsemble-3x1.1B-TinyMoE
](https://huggingface.co/JoPmt/TinyEnsemble-3x1.1B-TinyMoE)
> TinyEnsemble-3x1.1B-TinyMoE is a Mixture of Experts (MoE) made with the following models using LazyMergekit:
```
cognitivecomputations/TinyDolphin-2.8-1.1b
78health/TinyLlama_1.1B-function-calling
DaertML/TinyGauss-1.1B
```
Configuration
```
base_model: cognitivecomputations/TinyDolphin-2.8-1.1b
gate_mode: cheap_embed
dtype: bfloat16
experts:
  - source_model: cognitivecomputations/TinyDolphin-2.8-1.1b
    positive_prompts: ["write", "explain", "summarize", "how", "what", "acting"]
  - source_model: 78health/TinyLlama_1.1B-function-calling
    positive_prompts: ["code", "python", "javascript", "programming", "script", "run", "create"]
  - source_model: DaertML/TinyGauss-1.1B
    positive_prompts: ["count", "math", "algorithm", "crypto", "logic", "reason"]
```

<br><br>

- ### [Jamba-900M-GGUF](https://huggingface.co/Severian/Jamba-900M-GGUF/tree/main)
> original model from https://huggingface.co/pszemraj/jamba-900M-v0.13-KIx2

---

### TensorOpera-Fox-1-chat
Chat with web-based Document search and TensorOpera Fox-1 LlamaCPP

<img src='https://blog.tensoropera.ai/content/images/size/w1200/2024/06/fox-logo--1--3.jpg' height=400>


#### Description
project to talk with documents retrieved with websearch and enriched with newspaper3k
forced to use llama-cpp-python and not llamafile becuase of the embeddings

Using langchain for both llamaCPP LlamaCppEmbeddings and ChatLlamaCpp


#### MODEL USED: TensorOpera Fox-1
https://blog.tensoropera.ai/tensoropera-unveils-fox-foundation-model-a-pioneering-open-source-slm-leading-the-way-against-tech-giants/

We are thrilled to introduce TensorOpera Fox-1, our cutting-edge 1.6B parameter small language model (SLM) designed to advance scalability and ownership in the generative AI landscape. TensorOpera Fox-1 is a top-performing SLM in its class, outperforming SLMs developed by industry giants like Apple, Google, and Alibaba, making it an optimal choice for developers and enterprises looking for scalable and efficient AI deployment.

#### Create Venv
python311 -m venv venv
➜ venv\Scripts\activate
(venv) ➜ llamacpp-agents ⚡                                                                                             3.11.7

##### install dependencies
```
pip install --upgrade langchain langchain-community faiss-cpu tiktoken duckduckgo-search llama-cpp-python rich newspaper3k easygui lxml_html_clean streamlit
```

[GGUF MODEL
](https://huggingface.co/QuantFactory/Fox-1-1.6B-GGUF/tree/main)

##### RESOURCES:
https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/

https://python.langchain.com/v0.1/docs/integrations/text_embedding/llamacpp/

https://python.langchain.com/v0.2/docs/concepts/#documents

https://python.langchain.com/v0.1/docs/integrations/tools/ddg/

https://stackoverflow.com/questions/77782167/modulenotfounderror-no-module-named-langchain-openai

https://python.langchain.com/v0.2/docs/integrations/chat/openai/

---

<br><br>

