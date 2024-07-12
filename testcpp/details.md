### Details for small models

```

──────────────────────────────────────── Pythia-31M-Chat-v1 ────────────────────────────────────────
Model File: Pythia-31M-Chat-v1.Q8_0.gguf
───────────────────────────────────────────── Details ──────────────────────────────────────────────
Training Context Window: 2048
Stop Tokens: ['<|endoftext|>']
This model has Chat Template? True
────────────────────────────────────────────────────────────────────────────────────────────────────
                                           PROMPT FORMAT:


 <|im_start|>system
 {system_message}<|im_end|>
 <|im_start|>user
 {user_message}<|im_end|>
 <|im_start|>assistant


 Recommended inference parameters
 penalty_alpha: 0.5
 top_k: 2
 repetition_penalty: 1.0016

────────────────────────────────────────────────────────────────────────────────────────────────────
                                          Additional info:

 GGUF Repo: https://huggingface.co/Felladrin/gguf-Pythia-31M-Chat-v1/tree/main
 Original Repo: https://huggingface.co/Felladrin/Pythia-31M-Chat-v1
 Model architecture: gptneox
 Number of parameters: 31M

====================================================================================================



───────────────────────────────────────── Aira-2-124M-DPO ──────────────────────────────────────────
Model File: Aira-2-124M-DPO.Q8_0.gguf
───────────────────────────────────────────── Details ──────────────────────────────────────────────
Training Context Window: 1024
Stop Tokens: ['<|endofcompletion|>']
This model has Chat Template? False
────────────────────────────────────────────────────────────────────────────────────────────────────
                                           PROMPT FORMAT:


 <|startofinstruction|>What is a language model?<|endofinstruction|>
 A language model is a probability distribution over a vocabulary.<|endofcompletion|>

────────────────────────────────────────────────────────────────────────────────────────────────────
                                          Additional info:

 GGUF Repo: https://huggingface.co/Felladrin/gguf-Aira-2-124M-DPO
 Original Repo: https://huggingface.co/nicholasKluge/Aira-2-124M-DPO
 Model architecture: gpt2
 Number of parameters: 124M

====================================================================================================



───────────────────────────────────── smol-llama-101m-chat-v1 ──────────────────────────────────────
Model File: smol-llama-101m-chat-v1.q8_0.gguf
───────────────────────────────────────────── Details ──────────────────────────────────────────────
Training Context Window: 1024
Stop Tokens: ['</s>']
This model has Chat Template? True
────────────────────────────────────────────────────────────────────────────────────────────────────
                                           PROMPT FORMAT:


 <|im_start|>system
 {system_message}<|im_end|>
 <|im_start|>user
 {user_message}<|im_end|>
 <|im_start|>assistant

 Recommended inference parameters
 penalty_alpha: 0.5
 top_k: 4
 repetition_penalty: 1.105

────────────────────────────────────────────────────────────────────────────────────────────────────
                                          Additional info:

 GGUF Repo: https://huggingface.co/afrideva/Smol-Llama-101M-Chat-v1-GGUF
 Original Repo: https://huggingface.co/Felladrin/Smol-Llama-101M-Chat-v1
 Model architecture: llama
 Number of parameters: 101M

====================================================================================================




─────────────────────────────────────────── Aira-2-355M ────────────────────────────────────────────
Model File: Aira-2-355M.Q6_K.gguf
───────────────────────────────────────────── Details ──────────────────────────────────────────────
Training Context Window: 1024
Stop Tokens: ['<|endoftext|>']
This model has Chat Template? False
────────────────────────────────────────────────────────────────────────────────────────────────────
                                           PROMPT FORMAT:


 <|startofinstruction|>What is a language model?<|endofinstruction|>
 A language model is a probability distribution over a vocabulary.<|endofcompletion|>


 Recommended inference parameters
 penalty_alpha: 0.5
 top_k: 2
 repetition_penalty: 1.0016

────────────────────────────────────────────────────────────────────────────────────────────────────
                                          Additional info:

 GGUF Repo: https://huggingface.co/Felladrin/gguf-Aira-2-355M/tree/main
 Original Repo: https://huggingface.co/nicholasKluge/Aira-2-355M
 Model architecture: gpt2
 Number of parameters: 355M

====================================================================================================




────────────────────────────────────── stablelm-2-zephyr-1_6b ──────────────────────────────────────
Model File: stablelm-2-zephyr-1_6b-Q5_K_M.gguf
───────────────────────────────────────────── Details ──────────────────────────────────────────────
Training Context Window: 4096
Stop Tokens: ['<|endoftext|>']
This model has Chat Template? True
────────────────────────────────────────────────────────────────────────────────────────────────────
                                           PROMPT FORMAT:


 <|user|>
 {prompt}<|endoftext|>
 <|assistant|>

────────────────────────────────────────────────────────────────────────────────────────────────────
                                          Additional info:

 GGUF Repo: https://huggingface.co/second-state/stablelm-2-zephyr-1.6b-GGUF/tree/main
 Original Repo: https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b
 Model architecture: stablelm
 Number of parameters: 1.6b

====================================================================================================



────────────────────────────────────── h2o-danube2-1.8b-chat ───────────────────────────────────────
Model File: h2o-danube2-1.8b-chat-Q5_K_M.gguf
───────────────────────────────────────────── Details ──────────────────────────────────────────────
Training Context Window: 8192
Stop Tokens: ['</s>']
This model has Chat Template? True
────────────────────────────────────────────────────────────────────────────────────────────────────
                                           PROMPT FORMAT:


 <|prompt|>Why is drinking water so healthy?</s><|answer|>

────────────────────────────────────────────────────────────────────────────────────────────────────
                                          Additional info:

 GGUF Repo: https://huggingface.co/h2oai/h2o-danube2-1.8b-chat-GGUF
 Original Repo: https://huggingface.co/h2oai/h2o-danube2-1.8b-chat-GGUF
 Model architecture: llama
 Number of parameters: 1.8b

====================================================================================================





```
