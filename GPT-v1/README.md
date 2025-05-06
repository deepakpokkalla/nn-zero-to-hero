## Neural Networks: Zero to Hero Course by Andrej Karpathy

Learning neural networks replicating the code from scratch following the YouTube videos/lectures and Jupyter notebooks from the original repo. Original repository can be found at https://github.com/karpathy/nn-zero-to-hero/tree/master. 

---

**Setting up the environment and installing packages to run code in VSCode.**
- conda create -n nanogpt python==3.8
- pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
- pip install transformers==2.11.0 datasets==2.21.0
- pip install tiktoken wandb tqdm

---

**Lecture 7: Let's build GPT: from scratch, in code, spelled out. [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY)**

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

- For all other links see the video description.

---
**Reference Links**
- Tokenizers: 
    - [SentencePiece by Google](https://github.com/google/sentencepiece)
    - [tiktoken by OpenAI](https://github.com/openai/tiktoken)

