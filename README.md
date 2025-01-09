## Learning Neural Networks from Andrej Karpathy Videos

Andrej Karpathy has a "Neural Networks: Zero to Hero" video series. I am replicating the code from scratch following his YouTube videos/lectures from the original repo. Original repository can be found at https://github.com/karpathy/nn-zero-to-hero/tree/master. 

---

**Setting up the environment and installing packages to run code in VSCode.**
- conda create -n nanogpt python==3.8
- pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
- pip install transformers==2.11.0 datasets==2.21.0
- pip install tiktoken wandb tqdm

---

**Popular videos**

- [State of GPT - How to build a chat assistant?](https://www.youtube.com/watch?v=bZQun8Y4L2A)
- [Intro to Large Language Models (LLMs)](https://www.youtube.com/watch?v=zjkBMFhNj_g)

---

**GPT folder: Let's build GPT: from scratch, in code, spelled out.**

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY). For all other links see the video description.
- [Attention is all you need](https://arxiv.org/abs/1706.03762)

The lecture builds a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. GPT can be thought of as a next-word (token) predictor or a document (prompt) completer. The video explains *attention mechanism*, the core of the transformer architecture, in a intutive way. 

Here is a simple way to understand *attention mechanism*: 

Imagine you have a cat and, for fun, you keep a journal where you note all the places your cat sat during the day (unlikely, but bear with me). Now, you need to complete this sentence:

"The cat is sitting on the ____."

How would you figure out the answer, given options like "mat," "couch," "floor," "bed," or "chair"? Let's look at a few approaches:

- You could refer to your journal and count how often each place appears after "sitting on." The place with the highest count, say "mat," would be your best guess. While simple, this method ignores the context of the current sentence i.e, are we talking about cat or dog.

- Another approach might be to consider all the words in the sentence and average their "meanings", treating them equally without considering their importance or order. In simple terms, this means treating all the words equally without understanding which ones are more relevant. While this might give a rough guess, it fails to emphasize the context or relevance of specific words in the sentence.

- Now think about how you would naturally approach this. You would process the sentence by focusing on key elements—who we’re talking about ("cat") and what it’s doing ("sitting"). Not all words are equally important, and their positions in the sentence also matter. The attention mechanism captures this by calculating a weighted average of the information (values) from all words. The weights are determined based on how relevant each word (key) is to the word we’re focusing on (query). Words that are more important or closely related to the query get higher weights, while irrelevant ones get lower weights. This allows the model to focus on the right words and their relationships, enabling it to make accurate predictions based on context.

---

**GPT-2 folder: Let's reproduce GPT-2.**

- [YouTube video lecture](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10). For all other links see the video description.
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3 paper](https://arxiv.org/abs/2005.14165)
- [build-nanoGPT repo](https://github.com/karpathy/build-nanogpt). The specific original repository to reference while coding GPT-2 from scratch. 
- [nanoGPT repo](https://github.com/karpathy/nanoGPT)
- [llm.c repo](https://github.com/karpathy/llm.c)

---