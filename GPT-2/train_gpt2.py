from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import time

# -----------------------------------------
# Attention is a communication operation: 1024 tokens lined up. It's an aggregation/pooling/weighted sum function/reduce operation 
# MLP-FFN:each token is processed individually, acts as a "map"
# Transformer: Attention (reduce) + MLP (map); made-up opf nn.Linear and nn.Embedding layers!
# NOTE: self-attention is the communication. FeedForward is thinking/mapping on the data individually. 

# Instead of MultiHead and Head in GPT, single Self-attn class here!
# Attention mechanism steps:
# 1. 1024 tokens lined up in a sequence, emits query, key, value vectors. 
# 2. queries and keys get multiplied to get the attention amount/weight

class CausalSelfAttention(nn.Module):
    """ Multiple-heads are just parallel streams with their ouputs concatenated"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        # non-trainable attribute, available as self.bias
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd) or channels
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) or n_embd = nh*hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        
        # # attention (materializes the large(T,T) matrix for all the queries and keys)
        # # these operations take a ton of memory
        # att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1))) # head_size(hs) instead of C/n_embd here
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1) # (B, nh, T, T)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1,2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # GPT-2/BERT used 'tanh' approximate
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # layernorm along C dimension, per-token transformation
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # feed-forward network or FFN
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # x shape is # (B,T,C) where C is n_embd/channels dim
        x = x + self.mlp(self.ln_2(x)) # (B, T, C)
        return x

# GPT-2 (124M) Parameters
@dataclass
class GPTconfig:
    block_size: int = 1024 # max sequence length or context_length/block_size; T < block_size as this is MAX. value
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension (e.g., 384/n_head(6) = 64 (head_size))

class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), # additional layernorm after final self-attn block
        ))
        # biggest matrix multiplication: more time consuming 768 --> 50257
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme (uses twice in the forward pass)
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    # initialization as per GPT-2 source code
    # only other layer that requires init is nn.LayerNorm: PyTorch defaults are same as GPT-2, so no changes!
    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # pytorch default for bias is uniform distribution
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets = None): 
        # idx is of shape (B, T); T cannot be > block_size (max sequence length)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) # (B,T,n_embd) -> (B,T,n_embd)
        logits = self.lm_head(x) # (B,T,n_embd) -> (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),targets.view(-1))
        return logits, loss

    @classmethod # return the GPT object if we give model_type
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 mdel weights from hugginface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined  from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 124M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 124M params
        }[model_type]
        config_args["vocab_size"] = 50257 # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTconfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # ignore only attn bias -> auto-regressive masks
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create  optim groups. Any parameters that is 2D will be weight decayed, otherwise no. 
        # i.e., all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() <2]
        optim_groups = [
            {'params': decay_params, 'weight_decay':weight_decay},
            {'params': nodecay_params, 'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(F"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
# -----------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self,B,T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens)// (B*T)} batches") # single process sees all the data
        # print(f"1 epoch per process = {len(self.tokens)// (B*T*num_processes)} batches") # multiple processes: sees (1/num_procesess)*data

        # state 
        self.current_position = self.B * self.T * self.process_rank 
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B,T)
        self.current_position += B*T*self.num_processes
        if self.current_position + (B*T*self.num_processes+1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank 
        return x, y
# -----------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 4 GPU: --> 4 H100's
# torchrun --standalone --nproc_per_node=4 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a dpp run
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # Rank of the GPU on a single node: used in multi-node setting
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 #2**19, ~0.5M, in number of tokens
B = 16 # micro-batch size
T = 1024 # sequence length
assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

torch.set_float32_matmul_precision('high') # enable tf32 

# -----------------------------------
# Speedup stats on H100:
# NOTE: tok/sec is not 8x because of the GPU memory bandwidth
# FP32: ~500ms/iter, 35k tok/sec # original (sign, range, precision)
# TF32: ~200ms/iter, 75k tok/sec (precision reduced)
# FP32:  (reduced exponent range & precision), gradient scalers are needed as exp. range is reduced
# BF16: ~190ms/iter, 86k tok/sec (precision reduced)
# torch.compile: ~80ms/iter, 200k tok/sec
# flass attn: ~60ms/iter, 270k tok/sec (exact result as torch compile) 
# vocab_size (50304): ~55ms/iter, 300k tok/sec
# fused AdamW: ~53ms/iter, 305k tok/sec
# grad_accum: ~1720ms/iter, 305k tok/sec 
# -----------------------------------

# create model
# model = GPT.from_pretrained('gpt2')
# model = GPT(GPTconfig())
model = GPT(GPTconfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # not "ddp_rank"
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# learning rate scheduler: cosine scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50 
def get_lr(it):
    # 1. linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr* (it+1)/warmup_steps
    # 2. if it > lr_decay_iters, return min_lr
    if it > max_steps:
        return min_lr
    # 3. in between, use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio<=1 # 1 when it=1 (51th step)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimizer!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9,0.95), eps=1e-8) # works: loss=5.880 (seed not fixed)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x,y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but 
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right. 
        loss = loss / grad_accum_steps # <--- normalizing the loss for gradient accumulation as loss has "reduction=mean"
        loss_accum += loss.detach() # detach the tensor from the graph --> leaf node
        if ddp: # to make loss.backward() synchronize only the last micro_step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)
        loss.backward() # grads get accumulated in loss.backward() with "+="" 
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) # gradient clipping
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step:{step} | loss:{loss_accum.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)
# import code; code.interact(local=locals())

# ----------------

num_return_sequences = 5
max_length = 30

# prefix tokens 
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) #(8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5,8)
x = tokens.to(device)


# generate! right now x is (B, T) where B=5, T=8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T) -> (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:,-1,:] # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (hugginface pipeline default)
        # topk_probs here becomes (5,50), topk_indices is (5,50)
        # topk_indices correspond to indices in vocabulary to extract tokens
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probs
        ix = torch.multinomial(topk_probs,1) # (B,1), selects 1 index from 50
        # gather the corresponding indices
        xcol = torch.gather(topk_indices,-1,ix) # (B,1)
        # append to the sequence
        x = torch.cat((x,xcol),dim=1) 

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)




