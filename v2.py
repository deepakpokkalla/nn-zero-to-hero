import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 #64 #364 # how many independent sequence will we process in parallel?
block_size = 32 #256 #256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100 #500
learning_rate = 1e-3 #3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64 #384 # embedding dimension
n_head = 4 #6
n_layer = 4 #6
dropout = 0.0 #0.2
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
dfjhfdhs

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

# ===========================
# # OUTPUT OF THIS CODE # #

# step 0: train_loss 4.4116, val loss: 4.4022
# step 100: train_loss 2.6568, val loss: 2.6670
# step 200: train_loss 2.5091, val loss: 2.5058
# step 300: train_loss 2.4196, val loss: 2.4336
# step 400: train_loss 2.3503, val loss: 2.3565
# step 500: train_loss 2.2963, val loss: 2.3125
# step 600: train_loss 2.2411, val loss: 2.2504
# step 700: train_loss 2.2053, val loss: 2.2188
# step 800: train_loss 2.1640, val loss: 2.1874
# step 900: train_loss 2.1243, val loss: 2.1503
# step 1000: train_loss 2.1038, val loss: 2.1311
# step 1100: train_loss 2.0691, val loss: 2.1179
# step 1200: train_loss 2.0374, val loss: 2.0785
# step 1300: train_loss 2.0244, val loss: 2.0644
# step 1400: train_loss 1.9925, val loss: 2.0363
# step 1500: train_loss 1.9691, val loss: 2.0302
# step 1600: train_loss 1.9637, val loss: 2.0469
# step 1700: train_loss 1.9429, val loss: 2.0140
# step 1800: train_loss 1.9093, val loss: 1.9943
# step 1900: train_loss 1.9117, val loss: 1.9886
# step 2000: train_loss 1.8840, val loss: 1.9916
# step 2100: train_loss 1.8738, val loss: 1.9750
# step 2200: train_loss 1.8596, val loss: 1.9602
# step 2300: train_loss 1.8578, val loss: 1.9547
# step 2400: train_loss 1.8417, val loss: 1.9415
# step 2500: train_loss 1.8162, val loss: 1.9414
# step 2600: train_loss 1.8248, val loss: 1.9378
# step 2700: train_loss 1.8126, val loss: 1.9338
# step 2800: train_loss 1.8055, val loss: 1.9244
# step 2900: train_loss 1.8027, val loss: 1.9269
# step 3000: train_loss 1.7993, val loss: 1.9224
# step 3100: train_loss 1.7706, val loss: 1.9181
# step 3200: train_loss 1.7559, val loss: 1.9158
# step 3300: train_loss 1.7570, val loss: 1.9059
# step 3400: train_loss 1.7553, val loss: 1.8935
# step 3500: train_loss 1.7386, val loss: 1.8942
# step 3600: train_loss 1.7262, val loss: 1.8864
# step 3700: train_loss 1.7288, val loss: 1.8827
# step 3800: train_loss 1.7196, val loss: 1.8960
# step 3900: train_loss 1.7226, val loss: 1.8749
# step 4000: train_loss 1.7160, val loss: 1.8638
# step 4100: train_loss 1.7138, val loss: 1.8760
# step 4200: train_loss 1.7028, val loss: 1.8638
# step 4300: train_loss 1.7013, val loss: 1.8495
# step 4400: train_loss 1.7009, val loss: 1.8616
# step 4500: train_loss 1.6872, val loss: 1.8494
# step 4600: train_loss 1.6907, val loss: 1.8398
# step 4700: train_loss 1.6850, val loss: 1.8460
# step 4800: train_loss 1.6660, val loss: 1.8415
# step 4900: train_loss 1.6708, val loss: 1.8339
# step 4999: train_loss 1.6636, val loss: 1.8216

# And they bridle.

# ShALLOUS:
# Which thou wast were thrust the gatanss:
# Wart If usqual tothys dilth ane away, my fears's wize must off Latcfuldie but my would
# As egriededs, I in latist, drove to the vilt.
# Good you muselfrind that this me;
# But y prunt and plaw yet lets.
# In Badiet, and whom the sclittle
# Of alind to dry I by wear mysoman,
# The shire strook of his butte and thrust for treagint my monte inleous,
# To firsh? gry on.

# HENRY BOLINGS:
# You ardsabed.

# EDWARY:
# Ithous knear teas repts I crouf to young to too mary.
# You contrantymes have myse.-
# And fortwerle madany that such swon thee;
# Indent my courfessy stoutchs lord
# Is as deterp work.

# LUCIO:
# Aberlioves:
# Hed these ortess known, some stingd his refest, as not commpon;
# Mest is gon in Racce, movery I want bear
# that fravius wrants slevoss make more old lack.

# PRAMILLA:
# Way, subring.

# KING HENRY GAREY:
# Ceame,
# Second to meine the should e suble will the cliut.

# KING ROMENCENTENBRTANGELE:
# It do knew you, drawn I long, wout in not Glooding to ene runt,
# And you alsir, I wonce; gentle unclain,
# I car to take all notace, I have
# But be ressitions so me wortingin in--
# And great fatht did, my look break:
# Swear I have a canies to make the any in this blame,
# Would trupleing bromen is what sue would in thas jeth your Hrannertand.
# Not, our trancord, teat just defrined;
# Make his armies makes love is stay just torment:
# Slat you behorfs begut;
# Good vonby the town, must a be; I now;
# I the deceart om Twrieste you there,
# And so it out
# But ransuless thee, smetion.

# HERMORTIO:
# No, say, as If have to kavil scace:
# Out.

# PORALLINA:
# My flaring stray breath, yet, is shame, ar you
# forth, his
# 'Tward scome roudd. Prown, Garria give, fetter,
# Toh, sinstans o' the hath seed; command to lett,
# Go fly this fathar heave out cittuty art, lives?

# LARTENTES:
# O the womn'd,
# Talk'd our they will theirs is.
# Why, I world I ambed and was fatht not fantury,
# Pry prozens thy armsself, Joancet
# To comforn, Juriod parfority, wit shown's fortunds, sucher her the world an wi