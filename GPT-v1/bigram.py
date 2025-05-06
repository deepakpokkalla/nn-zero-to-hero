import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequence will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# --------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# tokenizer: create a mapping from characters to integers (character-level) "encoder - decoder"
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

# data loading i.e., dataloader
# generate a small batch of data of inputs x and targets y
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

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

# NOTE: In Bigram model, position encoding doesn't matter as - next token is predicted based only on the current token.
# Given this token, pluck the corresponding row from embedding matrix, compute probs via softmax on logits, and predict next token.  
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # because the emb_dim = vocab_size here. Thus, 
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
        # nn.Embedding.weight is the learnable weight matrix (num_embeddings/vocab_size, emb_dim)
    
    def forward(self,idx,targets=None):
        # idx and targets are both (B,T) tensor of integers
        # retrieve's idx-th row from the embedding matrix
        logits = self.token_embedding_table(idx) # idx of size (B,T) --> logits: (B, T, C)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T) # targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        # NOTE: there is not limit on T value (context lunch or so), thus you can keep appending new tokens
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx) # calls forward method in subclass of nn.Module; feeds every time-step
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

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

for iter in range(max_iters):

    # every one in a while evaulate the loss on train and val sets
    if iter % eval_interval == 0:
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
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))
