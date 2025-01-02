import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 #364 # how many independent sequence will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # embedding dimension
n_head = 6
n_layer = 6
dropout = 0.2
# --------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# tokenizer: create a mapping from characters to integers (character-level)
# lambda function - lambda arguments: expression
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 905 is train data
train_data = data[:n]
val_data = data[n:]

# data loading
# generate a small batch of data of inputs x and targets y
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

# average loss over multiple batches
@torch.no_grad() # decorator/context manager: reduces memory usage, improves performance
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

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,C); C = head_size here! else, head_size instead of C
        q = self.query(x) # (B,T,C); C = head_size here! else, head_size instead of C
        # compute attention scores ("affinities"); C = head_size here! else, 'self.head_size' instead of C below. 
        wei = q @ k.transpose(-2,-1) *C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        
        # assume block_size > T - that's why :T here? block_size = T = 8 for now! 
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        # wei = wei.masked_fill(self.tril == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C); C = head_size here! else, head_size instead of C
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # where is this proj in transformers paper?
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        # each communication channel in multi-head attention is smaller compared to single-head attention
        # similar to Group Convolutions

        # output of the self-attention itself
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,n_heads*(C/n_heads)) --> (B,T,C)

        # output of the projection
        out = self.dropout(self.proj(out))
        return out

# NOTE: self-attention is the communication. FeedForward is thinking on the data individually. 
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # why to multiply with 4 as in transformers paper?
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self,x): # per-token level
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like 
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # communication
        self.ffwd = FeedForward(n_embd) # computation (B,T,C)
        self.ln1 = nn.LayerNorm(n_embd) # layernorm along C dimension, per-token transformation
        self.ln2 = nn.LayerNorm(n_embd) # layernorm along C dimension, per-token transformation

    # Pre-norm formulation: layernorm on x before self-attention (different from transformers paper)
    def forward(self,x):
        x = x + self.sa(self.ln1(x)) # (B,T,C)
        x = x + self.ffwd(self.ln2(x)) # (B,T,C)
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # nn.Embedding.weight is the learnable weight matrix (num_embeddings/vocab_size, emb_dim)

        # vocab_size != emb_dim (n_embd)
        # token identity embedding: C is same as n_embd as C is extracted from x shape after embedding!
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) 
        # token position embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )

        self.lm_head = nn.Linear(n_embd, vocab_size)  

        ## single head
        # # self.sa_head = Head(n_embd) # head_size = n_embd (C) = 64

        ## multi-head, single block
        # self.sa_heads = MultiHeadAttention(4,n_embd//4) # i.e., 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedForward(n_embd)


    def forward(self,idx,targets=None):
        B, T = idx.shape # idx = xb (B, T)

        # idx and targets are both (B,T) tensor of integers
        # retrieve's idx-th row from the embedding matrix
        tok_emb = self.token_embedding_table(idx) # idx of (B,T) --> (B, T, C), where C = emb_dim
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C) - broadcasting of new dim along batch
        # feed it to self-attention head: that will apply softmax (ignore future tokens)
        # x = self.sa_heads(x) # apply one head of self-attention. (B, T, C) --> (B, T, C)
        # x = self.ffwd(x) # (B,T,C); per-token basis, independently. 

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T) # targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # NOTE:
    # Embedding 1: we have embedded the identity of the tokens
    # Embedding 2: Positional embedding for the tokens 
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        # NOTE: there is not limit on T value (context lunch or so), thus you can keep appending new tokens
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond) # calls forward method in subclass of nn.Module; feeds every time-step
            # print(logits.shape)
            # print(logits)
            # focus only on the last time step
            logits = logits[:,-1,:] # becomes (B,C) # looks at only last token in bigram model
            # print(logits.shape)
            # print(logits)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs,num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx,idx_next), dim=1) # (B,T+1)
            # print(idx.shape)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M paramters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

for iter in range(max_iters):

    # every one in a while evaulate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train_loss {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context,max_new_tokens=2000)[0].tolist()))


# # OUTPUT OF THIS CODE on H100 # # 

# 10.788929 M paramters
# step 0: train_loss 4.2849, val loss: 4.2823
# step 500: train_loss 1.6795, val loss: 1.8470
# step 1000: train_loss 1.3514, val loss: 1.5833
# step 1500: train_loss 1.2318, val loss: 1.5024
# step 2000: train_loss 1.1468, val loss: 1.4818
# step 2500: train_loss 1.0800, val loss: 1.4827
# step 3000: train_loss 1.0184, val loss: 1.5004
# step 3500: train_loss 0.9579, val loss: 1.5323
# step 4000: train_loss 0.8959, val loss: 1.5622
# step 4500: train_loss 0.8300, val loss: 1.5931
# step 4999: train_loss 0.7769, val loss: 1.6500
