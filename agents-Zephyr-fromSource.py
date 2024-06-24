#######################################################################
# Use Gradio API for 4 models:
# - DolphinLlama3_70bLLMChat
# - CasualLM_35bLLMChat
# - Mistral03LLMChat
# - Qwen1.5-72b_Chat
# - qwen1.5-MoE-A2.7B-Chat
#######################################################################
from rich.markdown import Markdown
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
console = Console(width=90)
from time import sleep
from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from gradio_client import Client
from gradio_client import Client
from langchain_core.prompts import PromptTemplate


def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

timestamp = datetime.datetime.now()
log = f"""




#########################################################################
{str(timestamp)}

Using Mistralv0.3 steps 1-3
Using Qwen-72b for step 4
#########################################################################

"""
writehistory('history.txt',log)


class GradioLLMChat(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use Qwen/Qwen2-72B-Instruct from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chatbot = Client("Qwen/Qwen2-72B-Instruct")

    @property
    def _llm_type(self) -> str:
        return "Gradio API client Qwen/Qwen2-72B-Instruct"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client Meta_llama3_8b using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                param,	# str  in 'Input' Textbox component
                [],	# Tuple[str | Dict(file: filepath, alt_text: str | None) | None, str | Dict(file: filepath, alt_text: str | None) | None]  in 'Qwen1.5-72B-Chat' Chatbot component
                prompt,	# str  in 'parameter_9' Textbox component
                api_name="/model_chat"
        )
        return result[1][0][1]
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        chatbot=None,
        request: float = 0.95,
        param: float = 512,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        if chatbot is None:
            chatbot = self.chatbot
        # Return the response from the API
        for char in chatbot.submit(   #.submit for streaming effect / .predict for normal output
                param,	# str  in 'Input' Textbox component
                [],	# Tuple[str | Dict(file: filepath, alt_text: str | None) | None, str | Dict(file: filepath, alt_text: str | None) | None]  in 'Qwen1.5-72B-Chat' Chatbot component
                prompt,	# str  in 'parameter_9' Textbox component
                api_name="/model_chat"
                )[1][0][1]:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk





from langchain_core.language_models.llms import LLM
class DolphinLlama3_70bLLMChat(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use Qwen 1.5-72b-chat from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        yourHFtoken = "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" #here your HF token
        self.chatbot = Client("cognitivecomputations/chat", hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client dolphinLlama70b"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client dolphinLlama70b using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                message=  prompt,   #"Why you should learn AI?",
                request="dolphin-2.9.1-llama-3-70b.Q3_K_M.gguf",
                param_3=8192,
                param_4=0.1,
                param_5=0.95,
                param_6=40,
                param_7=1.3,
		            api_name="/chat"                               
        )
        return result



from langchain_core.language_models.llms import LLM
class MetaLlama3_8bLLMChat(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use Llama3-8b-chat from HF Spaces API
    https://huggingface.co/spaces/gnumanth/llama3-chat
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        yourHFtoken = "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" #here your HF token
        self.chatbot = Client("gnumanth/llama3-chat", hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client gnumanth/llama3-chat"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client gnumanth/llama3-chat using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                message=  prompt,   #"Why you should learn AI?",
                request="You are a smart writer assistant", #system Message
                param_3=4096, #max new tokens
                param_4=0.25, #temperature, can also be 0
                api_name="/chat"                         
                    )
        return result

class CasualLM_35bLLMChat(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use Qwen 1.5-72b-chat from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #yourHFtoken = "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" #here your HF token
        self.chatbot = Client("JosephusCheung/CausalLM-35B-long-Q6K-GGUF") #, hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client dolphinLlama70b"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client dolphinLlama70b using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                                  message=prompt,
                                  request='you are a helpful assistant',
                                  param_3=4096,
                                  param_4=0.7,
                                  param_5=0.95,
                                  param_6=40,
                                  param_7=1.3,
                                  param_8="35b-beta-long-Q6_K.gguf",
                                  api_name="/chat"
        )
        return result



class Mistral03LLMChat(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use Mistral_V0.3 from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #yourHFtoken = "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" #here your HF token
        self.chatbot = Client("ysharma/Chat_with_Mistral_V0.3") #, hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client dolphinLlama70b"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client Mistral_V0.3 using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                                  message=prompt,
                                  request=0.35,
                                  param_3=4096,
                                  api_name="/chat"
                                      )
        return result


class Mixtral(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use Ale6100/mistralai-Mixtral-8x7B-Instruct-v0.1 from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #yourHFtoken = "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" #here your HF token
        self.chatbot = Client("Ale6100/mistralai-Mixtral-8x7B-Instruct-v0.1") #, hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client Ale6100/mistralai-Mixtral-8x7B-Instruct-v0.1"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client Ale6100/mistralai-Mixtral-8x7B-Instruct-v0.1 using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                                  param_0=prompt,
                                  api_name="/predict"
                                      )
        return result



class Phi3Mini_128k(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use eswardivi/Phi-3-mini-128k-instruct from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #yourHFtoken = "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" #here your HF token
        self.chatbot = Client("eswardivi/Phi-3-mini-128k-instruct") #, hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client eswardivi/Phi-3-mini-128k-instruct"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client Mistral_V0.3 using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                                  message=prompt,
                                  request=0.35,
                                  param_3=True,
                                  param_4=2000,
                                  api_name="/chat"
                                      )
        return result


class Phi3Medium_128k(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use rombodawg/Phi-3-medium-128k-instruct from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #yourHFtoken = "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" #here your HF token
        self.chatbot = Client("rombodawg/Phi-3-medium-128k-instruct") #, hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client rombodawg/Phi-3-medium-128k-instruct"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client rombodawg/Phi-3-medium-128k-instruct using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                                message=prompt,
                                request=0.35,
                                param_3=4096,
                                param_4=0.7,
                                param_5=40,
                                api_name="/chat"
                                      )
        return result



class Zephr(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use sablab/Zephyr from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #yourHFtoken = "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" #here your HF token
        self.chatbot = Client("sablab/Zephyr") #, hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client sablab/Zephyr"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client sablab/Zephyr using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                                message=prompt,
                                system_message="",
                                max_tokens=2048,
                                temperature=0.4,
                                top_p=0.95,
                                api_name="/chat"
                                      )
        return result



class CasualLM14b(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use sJosephusCheung/CausalLM-14B-DPO-GGUF from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #yourHFtoken = "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" #here your HF token
        self.chatbot = Client("JosephusCheung/CausalLM-14B-DPO-GGUF") #, hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client JosephusCheung/CausalLM-14B-DPO-GGUF"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client JosephusCheung/CausalLM-14B-DPO-GGUF using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                                message=prompt,
                                request="",
                                param_3=2048,
                                param_4=0.7,
                                param_5=0.95,
                                param_6=40,
                                param_7=1.2,
                                param_8="causallm_14b-dpo-alpha.f16.gguf",
                                api_name="/chat"
                                      )
        return result

class Llama3_70b(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use Be-Bo/llama-3-chatbot_70b from HF Spaces API
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #yourHFtoken = "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" #here your HF token
        self.chatbot = Client("Be-Bo/llama-3-chatbot_70b") #, hf_token=yourHFtoken)

    @property
    def _llm_type(self) -> str:
        return "Gradio API client Be-Bo/llama-3-chatbot_70b"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client Be-Bo/llama-3-chatbot_70b using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                                message=prompt,
                                api_name="/chat"
                                      )
        return result


class QwenMoE2LLMChat(LLM):
    """
    Custom LLM class based on the Gradio API call.
    It will use Qwen/qwen1.5-MoE-A2.7B-Chat-demo from HF Spaces API
    Qwen/Qwen2-57b-a14b-instruct-demo
    client = Client("Qwen/Qwen2-57b-a14b-instruct-demo")
        result = client.predict(
                query="Hello!!",
                history=[],
                system="You are a helpful assistant.",
                api_name="/model_chat"
    """
    from typing import Any, Dict, Iterator, List, Mapping, Optional
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.outputs import GenerationChunk
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #yourHFtoken = "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" #here your HF token
        self.chatbot = Client("Qwen/Qwen2-57b-a14b-instruct-demo") #,hf_token=yourHFtoken

    @property
    def _llm_type(self) -> str:
        return "Gradio API client Qwen/qwen1.5-MoE-A2.7B-Chat-demo"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95, #it's the history touple
            param: str = 'You are a helpful assistant', #it's the system message
    ) -> str:
        """
        Make an API call to the Gradio API client Qwen/qwen1.5-MoE-A2.7B-Chat-demo using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
                query=  prompt,   #"Why you should learn AI?",
                history=[],
                system="You are a smart writer assistant", #system Message
                api_name="/model_chat"                           
                    )
              
        return result[1][0][1]
    
from gradio_client import Client
from langchain_core.prompts import PromptTemplate

def countTokens(text):
  import tiktoken
  encoding = tiktoken.get_encoding("r50k_base")
  context_count = len(encoding.encode(text))
  return context_count



def firstOutliner(llm, content):
  outliner_template = PromptTemplate.from_template("""Write an outline for a long Medium Article about the following topic:
---
{topic}
---

Here is the format of your writing:
1. Use "#" Title " to indicate section title , "##" Title " to indicate subsection title , "###" Title " to indicate subsubsection title , and so on.
2. Do not include other information.
""")
  prompt = outliner_template.invoke({"topic": content}).text
  res = llm.invoke(prompt)
  return res

def improvedOutliner(llm, firstoutline):
  outliner_template = PromptTemplate.from_template("""Improve an outline for a long Medium Article.
You already have a draft outline that covers the general information. Now you want to improve it to make it more comprehensive.

[start of draft outline]
{topic}
[end of draft outline]

Here is the format of your writing:
1. Use "#" Title " to indicate section title , "##" Title " to indicate subsection title , "###" Title " to indicate subsubsection title , and so on.
2. Do not include other information.
""")
  prompt = outliner_template.invoke({"topic": firstoutline}).text
  res = llm.invoke(prompt)
  return res

def _2_writeAgent(llm, structuredoutline):
  write_template = PromptTemplate.from_template('''You are an expert writer assistant tasked with writing excellent articles for Medium, starting from an existing content.
write the best 2500 words article using the provided title, subheading and existing contents. 
Always include the section titles: drescribe in details and organize the paraghraphs following the outline, adding meaningful content. 

[start of structured outline]
{structured_outline}
[end of structured outline]

Use the following writing style.
write with Technical expertise demonstrating a deep understanding of the technical aspects of the field. Uses specific terminology and explains complex concepts in a clear and concise manner.
write with a Storytelling approach: incorporates anecdotes and personal experiences into his articles, giving them a relatable and engaging quality. Shares lessons learned from professional life, which adds depth and authenticity to the writing.
Use Metaphors and illustrations: use metaphors (e.g., "final element" as a metaphor for decision-making) to explain abstract ideas or processes, making them easier to grasp for readers.
Add a Reflective tone: the writing must also have a reflective quality, discussing personal growth, decision-making, and the importance of critical thinking. Encourage the readers to learn from their experiences and apply that learning to their lives and work.
Include a Call to action: urge the readers to take responsibility for their decision-making, choose their next steps wisely, or adopt new strategies for managing emotions and learning new topics.
Overall use a Conversational style: Despite the technical content, maintain a conversational tone, making it accessible to a wider audience.
Blending personal and professional: seamlessly combines personal stories with professional insights, creating a distinctive voice that is both informative and relatable.
Focus on human aspects: While discussing technical subjects, emphasize the human element, such as decision-making, emotional intelligence, and effective communication.
Use of memorable phrases: employ catchy phrases that stick with the reader and encapsulate his main points.
Show, don't tell: Use descriptive language to paint a picture with words.
 
Your Essay:
''')

  prompt = write_template.invoke({'structured_outline':structuredoutline}).text
  res = llm.invoke(prompt)
  return res

######################################################################################
def ReflectionAgent(llm, content):
  reflection_prompt = PromptTemplate.from_template('''You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
Provide a grade and detailed recommendations, including requests for length, depth, style, etc.

{request}
''')
  prompt = reflection_prompt.invoke({"request": content}).text
  res = llm.invoke(prompt)
  return res


def RedactionAgent(llm, essay, critiques):
  redactionPrompt = PromptTemplate.from_template('''You are an expert writer. Starting from an essay and a critique review, write a new article and apply the remarks and recommendations from the critiques.

[start of original essay]                                                
{essay}
[end of original essay]
                                                 
[start of critique review]                                               
{critiques}
[end of critique review] 
                                                 
Use the following writing style.
write with Technical expertise demonstrating a deep understanding of the technical aspects of the field. Uses specific terminology and explains complex concepts in a clear and concise manner.
write with a Storytelling approach: incorporates anecdotes and personal experiences into his articles, giving them a relatable and engaging quality. Shares lessons learned from professional life, which adds depth and authenticity to the writing.
Use Metaphors and illustrations: use metaphors (e.g., "final element" as a metaphor for decision-making) to explain abstract ideas or processes, making them easier to grasp for readers.
Add a Reflective tone: the writing must also have a reflective quality, discussing personal growth, decision-making, and the importance of critical thinking. Encourage the readers to learn from their experiences and apply that learning to their lives and work.
Include a Call to action: urge the readers to take responsibility for their decision-making, choose their next steps wisely, or adopt new strategies for managing emotions and learning new topics.
Overall use a Conversational style: Despite the technical content, maintain a conversational tone, making it accessible to a wider audience.
Blending personal and professional: seamlessly combines personal stories with professional insights, creating a distinctive voice that is both informative and relatable.
Focus on human aspects: While discussing technical subjects, emphasize the human element, such as decision-making, emotional intelligence, and effective communication.
Use of memorable phrases: employ catchy phrases that stick with the reader and encapsulate his main points.
Show, don't tell: Use descriptive language to paint a picture with words.
                                                 
The new article:
''')
  prompt = redactionPrompt.invoke({"essay": essay, "critiques": critiques}).text
  res = llm.invoke(prompt)
  return res
#######################################################################################


def MYStyleAgent(llm, article):
  restylePrompt = PromptTemplate.from_template('''You are an expert writer. Given a provided article, rewrite it into an engaging article.
Organize the content with clarity.                                               

[start of provided article]
{article}
[end of provided article]

Use the following writing style.
write with Technical expertise demonstrating a deep understanding of the technical aspects of the field. Uses specific terminology and explains complex concepts in a clear and concise manner.
Use a Storytelling approach: incorporates anecdotes and personal experiences into his articles, giving them a relatable and engaging quality. 
Overall use a Conversational style: Despite the technical content, maintain a conversational tone, making it accessible to a wider audience.
Blending personal and professional: seamlessly combines personal stories with professional insights, creating a distinctive voice that is both informative and relatable.
Focus on human aspects: While discussing technical subjects, emphasize the human element, such as decision-making, emotional intelligence, and effective communication.
Use of memorable phrases: employ catchy phrases that stick with the reader and encapsulate his main points.
Show, don't tell: Use descriptive language to paint a picture with words.
 
Your Article:
''')
  prompt = restylePrompt.invoke({"article": article}).text
  res = llm.invoke(prompt)
  return res


correctedArticle = """title: Progress towards true Artificial General Intelligence (AGI) has hit a wall.
subheading: Most AI benchmarks measure skill. But skill is not intelligence. What can be done?

Article: ### The stalling of AGI (Artificial General Intelligence) progress and the need for new ideas.
The stalling of AGI (Artificial General Intelligence) progress and the need for new ideas can be attributed to several factors. 
1 - The reliance on memorization rather than reasoning has limited the capabilities of modern AI systems. These Language Large Models (LLMs) are adept at memorizing patterns in their training data and applying them in adjacent contexts, but they lack the ability to generate new reasoning based on novel situations. More training data may improve performance on memorization-based benchmarks, but this is not indicative of general intelligence. General intelligence requires the ability to efficiently acquire new skills, which is currently beyond the capabilities of LLMs.
2 - The inability of AI systems to generalize beyond their training data has hindered progress in AGI. While AI systems have excelled in specific tasks such as beating humans at games like poker, chess, and go, they have struggled to transfer their knowledge to other domains. This failure to generalize means that AI will always be limited by the human general intelligence in the loop. To truly achieve AGI, we need systems that can discover and invent alongside humans, pushing humanity forward.
3 - The lack of transparency and collaboration in the field of AI has slowed progress. With the release of GPT-4, frontier AGI progress has become closed source, with technical details kept under wraps for "competitive" reasons. This trend has shifted research attention away from new architectures and algorithms, and towards scaling existing models. The belief that "scale is all you need" has become pervasive, despite the fact that it is not supported by evidence. Moreover, the closed nature of AI research has led to a concentration of resources in a few large companies, further limiting progress.
4 - The emphasis on benchmarking and competition has stifled creativity and innovation. AI benchmarks have rapidly saturated to human performance levels, leading to a focus on incremental improvements rather than radical breakthroughs. This has resulted in a narrow view of what constitutes progress in AI, with a focus on narrow tasks rather than general intelligence. Additionally, the lack of diversity in the field of AI has contributed to this problem, with certain perspectives and approaches dominating the discourse.
The lack of funding and support for open-source AGI research has hindered progress. While the internet and open source have been powerful innovation engines in the past, the current trend towards closed-source research has made it difficult for new ideas to emerge. This has led to a concentration of resources in a few large companies, further limiting progress. To overcome these challenges, we need to incentivize new ideas, promote collaboration and transparency, and support open-source research. Only then can we hope to make significant progress towards AGI.

### The limitations of modern AI (Language Large Models - LLMs) in terms of memorization and inability to generate new reasoning based on novel situations.
The limitations of modern AI (Language Large Models - LLMs) in terms of memorization and inability to generate new reasoning based on novel situations are a major obstacle to achieving artificial general intelligence (AGI). LLMs are adept at memorizing high-dimensional patterns in their training data and applying them in adjacent contexts, but they lack the ability to generate new reasoning based on novel situations. This is due to the fact that LLMs rely on memorization rather than reasoning, and they cannot generate new reasoning based on novel situations.
One of the main issues with LLMs is that they are limited by the amount of training data available to them. More training data can improve performance on memorization-based benchmarks, but it does not necessarily lead to general intelligence. General intelligence requires the ability to efficiently acquire new skills, which is currently beyond the capabilities of LLMs.
Another issue with LLMs is that they are unable to generalize beyond their training data. This means that they are unable to transfer their knowledge to new situations or tasks. For example, an AI system that has been trained to play chess may be able to beat humans at chess, but it would not be able to transfer its knowledge to other board games like checkers or Go.
The inability of LLMs to generalize is a major obstacle to achieving AGI. Without the ability to generalize, AI systems will always be limited by the human general intelligence in the loop. To truly achieve AGI, we need systems that can discover and invent alongside humans, pushing humanity forward.
Despite the success of LLMs in recent years, there is still much work to be done in the field of AI. The limitations of LLMs in terms of memorization and inability to generate new reasoning based on novel situations highlight the need for new ideas and approaches. By promoting open-source research and incentivizing new ideas, we can increase the rate of progress towards AGI and ensure that new ideas are widely distributed, establishing a more even playing field between small and large AI companies.

### The ARC AGI evaluation (ARC-AGI) and the need for open-source AGI research to increase the rate of new ideas and decrease the chances of AGI being discovered by a single lab.
The ARC AGI evaluation (ARC-AGI) is a crucial tool for measuring the progress of artificial general intelligence (AGI) and identifying the need for open-source AGI research. ARC-AGI is the only evaluation method that measures general intelligence, which is the ability to efficiently acquire new skills and solve novel, open-ended problems. The current state-of-the-art (SOTA) high score for ARC-AGI is only 34%, which highlights the significant gap between current AI systems and human intelligence. Given that humans can master tasks quickly, ARC-AGI is easy for humans and impossible for modern AI systems.
The need for open-source AGI research is crucial to increase the rate of new ideas and decrease the chances of AGI being discovered by a single lab. The current trend in AI research is moving towards closed-source research, which limits the sharing of ideas and knowledge. This trend is driven by the belief that "scale is all you need" and the desire to protect competitive advantages. However, this approach stifles innovation and limits the rate of progress towards AGI.
Open-source research, on the other hand, promotes collaboration and knowledge sharing, which accelerates the rate of progress towards AGI. By making research openly accessible, researchers from around the world can contribute to the development of new ideas and innovations. This approach also ensures that new ideas are widely distributed, establishing a more even playing field between small and large AI companies.
Moreover, the lack of transparency and collaboration in AI research is contributing to the misperception that AGI is imminent, which is influencing the AI regulatory environment. Regulators are considering roadblocks to frontier AI research under the wrong assumption that AGI is imminent. However, the truth is that no one knows how to build AGI. By promoting open-source research, we can accelerate the rate of progress towards AGI and ensure that new ideas are widely distributed, establishing a more even playing field between small and large AI companies.
In conclusion, the ARC AGI evaluation (ARC-AGI) is a crucial tool for measuring the progress of AGI and identifying the need for open-source AGI research. Open-source research promotes collaboration and knowledge sharing, which accelerates the rate of progress towards AGI and ensures that new ideas are widely distributed. By incentivizing open-source research, we can increase the rate of new ideas and decrease the chances of AGI being discovered by a single lab, ultimately leading to a more even playing field between small and large AI companies.

The paper "On the Measure of Intelligence" by Francois Chollet discusses the problem of measuring intelligence in machines and proposes a new evaluation called the ARC evaluation (ARC-E). The author argues that current evaluation methods, which focus on narrow tasks and do not account for general intelligence, are inadequate for measuring the true intelligence of machines.
Chollet defines general intelligence as the ability to acquire new skills through learning and experience, and to apply those skills in new and unfamiliar situations. He notes that current AI systems, including large language models (LLMs), are good at memorizing information and performing well on specific tasks, but they lack the ability to generalize and apply their knowledge to new situations.
To address this problem, Chollet proposes the ARC-E, which is designed to measure a system's ability to learn and generalize across a wide range of tasks. The evaluation consists of a series of questions that require a system to reason and think creatively, rather than simply recall information. The questions are designed to be challenging and require the system to use a variety of cognitive skills, such as pattern recognition, analogy-making, and causal reasoning.
Chollet argues that the ARC-E provides a more accurate measure of a system's general intelligence than current evaluation methods, and that it can help guide the development of more intelligent AI systems. He suggests that the ARC-E could be used to evaluate the intelligence of both narrow AI systems and more general AI systems, and that it could help researchers identify areas where current AI systems are lacking in intelligence.
Overall, the paper "On the Measure of Intelligence" presents a new evaluation method for measuring the intelligence of AI systems, and argues that this method is more effective than current evaluation methods for measuring general intelligence. The proposed evaluation, the ARC-E, is designed to challenge AI systems to think creatively and apply their knowledge in new situations, providing a more accurate measure of their true intelligence.

The ARC challenge, also known as the Abstraction and Reasoning Corpus (ARC-AGI), is a benchmark introduced by François Chollet in his influential paper "On the Measure of Intelligence" in 2019. The purpose of the ARC challenge is to measure the efficiency of AI skill-acquisition on unknown tasks, specifically focusing on general intelligence.
The ARC challenge consists of unique training and evaluation tasks, each containing input-output examples presented in the form of grids with squares of ten different colors. The goal is to produce a pixel-perfect correct output grid for the final output, including picking the correct dimensions of the output grid. The challenge is explicitly designed to compare artificial intelligence with human intelligence by listing the priors knowledge that humans possess, such as objectness, goal-directedness, numbers and counting, and basic geometry and topology.
The ARC challenge is considered the only AI benchmark that measures progress towards general intelligence, and solving it would represent a significant stepping stone towards AGI. It would result in a new programming paradigm that would allow anyone, even those without programming knowledge, to create programs simply by providing a few input-output examples of what they want. This would dramatically expand who is able to leverage software and automation, and programs could automatically refine themselves when exposed to new data, similar to how humans learn.
The ARC challenge has a history of competitions, starting with the first ARC-AGI competition on Kaggle in 2020, followed by the ARCathon 2022 and ARCathon 2023, with the most recent being the ARC Prize 2024 with a prize pool of over $1.1M."""


# GradioLLMChat(LLM)      It will use Qwen/Qwen2-72B-Instruct from HF Spaces API
# Mistral03LLMChat(LLM)   ysharma/Chat_with_Mistral_V0.3
# MetaLlama3_8bLLMChat(LLM)    It will use Llama3-8b-chat
# CasualLM_35bLLMChat(LLM) 
# DolphinLlama3_70bLLMChat(LLM)  dolphin-2.9.1-llama-3-70b.Q3_K_M.gguf
# QwenMoE2LLMChat(LLM)   Qwen2-57b-a14b-instruct-demo
#llm = Mistral03LLMChat()
#llm = GradioLLMChat()
#llm = Phi3Mini_128k()
#llm = Phi3Medium_128k()
#llm = Mixtral()
#llm = Zephr()
#CasualLM14b()


#console.rule('[bold bright_red]6.STYLED FINAL corrected ARTICLE')
#finalArticle = MYStyleAgent(llm,correctedArticle )
# this part is removed


#del llm
llm = Zephr() #Llama3_70b()  #Phi3Medium_128k() #QwenMoE2LLMChat()
#llm = CasualLM14b()
console.rule('[bold bright_red]4.CRITIQUE ON THE ARTICLE')
reflections = ReflectionAgent(llm,correctedArticle )  #finalArticle
console.print(reflections)
console.print(f"[green1]length OUTLINE = {countTokens(reflections)}")
console.print('[bold bright_red]#########################################################################')
log = f"""-----------4.CRITIQUE ON THE ARTICLE----------------
Critiques and reflections on the article... 
---

{reflections}
length article = {countTokens(reflections)}"
-----------------------------// //-------------------------------------

"""
writehistory('history.txt',log)

console.rule('[bold bright_red]5.APPLY CRITIQUEs ON THE ARTICLE')
correctedArticle = RedactionAgent(llm,correctedArticle,reflections )
console.print(correctedArticle)
console.print(f"[green1]length OUTLINE = {countTokens(correctedArticle)}")
console.print('[bold bright_red]#########################################################################')
log = f"""-----------5.APPLY CRITIQUEs ON THE ARTICLE----------------
Applying Critiques... 
---

{correctedArticle}
length article = {countTokens(correctedArticle)}"
-----------------------------// //-------------------------------------

"""
writehistory('history.txt',log)