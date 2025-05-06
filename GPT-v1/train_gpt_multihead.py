import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 #32 # how many independent sequences will we process in parallel?
block_size = 256 #8 # what is the maximum context length for predictions? T here!
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 #1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 #32 # embedding dimension
n_head = 6 #4
n_layer = 6 #4
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# tokenizer: create a mapping from characters to integers (character-level) "encoder - decoder"
# lambda function - lambda arguments: expression
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y # (batch_size, block_size) or (B, T)

# compute average loss over multiple batches: less noisy loss 
@torch.no_grad() # decorator/context manager: reduces memory usage as no gradient computation, improves performance (no backward calls)
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # (n_embd, hs)
        self.query = nn.Linear(n_embd, head_size, bias=False) # (n_embd, hs)
        self.value = nn.Linear(n_embd, head_size, bias=False) # (n_embd, hs)
        # non-trainable attribute, available as self.tril
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels or n_embd)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape # "C = head_size (hs)" here!
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # assume block_size > T - that's why :T here? block_size = T = 8 for now! 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # wei = wei.masked_fill(self.tril == 0, float('-inf')) # (B, T, T)
        wei = wei.softmax(dim=-1) # (B,T,T)
        wei = self.dropout(wei) # (B,T,T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs); Here, it's (B, T, C) as hs = (n_embd or C) in single-head attention
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # where is this proj in transformers paper? (shown in Fig.2, but in Fig.1 shown outside the block, why?)
        self.proj = nn.Linear(n_embd, n_embd) # n_embd = head_size * num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # each communication channel in multi-head attention is smaller compared to single-head attention
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, num_heads*(n_embd/head_size)) --> (B, T, n_embd)
        out = self.dropout(self.proj(out)) # (B, T, n_embd) 
        return out

# Self-attention does the communication. FeedForward is per-token level, thinking on the data individually. 
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # d_ff (ff inner dimension) is 4*n_embd as per transformers paper (Sec 3.3)
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # 4* adds some computation & grows the layer. 
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) # (B, T, n_embd)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # communication with n_head's & head_size; # i.e., 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd) # computation (B, T, n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # layernorm along n_embd dimension, per-token transformation
        self.ln2 = nn.LayerNorm(n_embd) # layernorm along n_embd dimension, per-token transformation

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # (B, T, n_embd)
        x = x + self.ffwd(self.ln2(x)) # (B, T, n_embd)
        return x

# single self-attention head model 
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # nn.Embedding.weight is the learnable weight matrix (num_embeddings/vocab_size, emb_dim)

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # token identity embedding, where "n_embd != vocab_size" unlike bigram model
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # token position embedding; (T, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # language modeling head; (n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape # idx = xb; shape: (B, T)

        # idx and targets are both (B,T) tensor of integers
        # retrieve's idx-th row from the embedding matrix/table to get "channel" dim or "n_embd"
        tok_emb = self.token_embedding_table(idx) # idx of size (B,T) --> logits: (B, T, n_embd); "n_embd = C" here!
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd); "n_embd = C" here!
        x = tok_emb + pos_emb # broadcasting (B, T, n_embd) + (T, n_embd) --> (B, T, n_embd), "n_embd = C" here!
        # feed it ("x" or encoded inputs) to self-attention head(s): that will apply softmax (ignore future tokens)
        # n_embd is passed to "sa_head" and extracted as C (channel dim) in forward of Head. Thus, "C = n_embd".
        x = self.blocks(x) # (B, T, n_embd), "n_embd = C" here!
        x = self.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, n_embd) --> (B,T,vocab_size)
        
        # C here is vocab_size 
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Embedding 1: we have embedded the identity of the tokens
    # Embedding 2: Positional embedding for the tokens 
    # C refers to vocab_size in logits during loss computation or token generation
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # NOTE: there is no limit on T value (context length or overall seq thus far), so you can keep appending new tokens.
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, T)
            # get the predictions
            logits, loss = self(idx_cond) # calls forward method in subclass of nn.Module; feeds every time-step
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, T, C/vocab_size) --> (B, C) # looks at only last token in bigram model
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C/vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train') # (B,T) both Xb & Yb

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))