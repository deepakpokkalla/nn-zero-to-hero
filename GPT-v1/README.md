## Lecture 7: Let's build GPT: from scratch, in code, spelled out.

Replicating a Generatively Pretrained Transformer (GPT), following the [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy, based on the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. The lecture talks about connections to ChatGPT, which has taken the world by storm. It also demonstrates GitHub Copilot, a GPT itself, that helps us write a GPT! It's recommended to watch the earlier [makemore series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) videos by Karpathy to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which are taken for granted in this GPT video.

- [GitHub repo for the video](https://github.com/karpathy/ng-video-lecture)
- [Google colab for the video](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
- [Attention is All You Need paper](https://arxiv.org/abs/1706.03762)

- [NanoGPT repo](https://github.com/karpathy/nanoGPT)
- [nn-zero-to-hero Github repo](https://github.com/karpathy/nn-zero-to-hero/tree/master)
- [nn-zero-to-hero video series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

---

**Environment setup**
- conda create -n nanogpt python==3.8
- pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
- pip install transformers==2.11.0 datasets==2.21.0
- pip install tiktoken wandb tqdm

---
**References**
- Tokenizers: 
    - [SentencePiece by Google](https://github.com/google/sentencepiece)
    - [tiktoken by OpenAI](https://github.com/openai/tiktoken)

