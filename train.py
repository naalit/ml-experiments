import torch
import time
import grc

batch_size = 64 # T: 64
block_size = 256 # T: 256; context length
max_iters = 7500
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # T: 384
n_heads = 6  # T: 6
n_layers = 6
dropout = 0.2
cache_ratio = 0.5

# --

torch.manual_seed(1337)

print("Using device", device)
torch.set_float32_matmul_precision("medium")

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("Start of text: ", text[:10])

# all unique characters in the dataset, sorted
chars = sorted(list(set(text)))
vocab_size = len(chars)

# -- encoder
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # str -> [int]
decode = lambda l: ''.join([itos[i] for i in l]) # [int] -> str

# -- torch
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# --
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
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

# --
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size, n_embd=n_embd, mask=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.mask = mask
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)     # (B,T,H)
        q = self.query(x)   # (B,T,H)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,H) @ (B,H,T) --> (B,T,T)
        if self.mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        v = self.value(x) # (B,T,H)
        out = wei @ v # (B,T,T) @ (B,T,H) --> (B,T,H)
        return out


class GRCHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.cache_dim = head_size # int(head_size*cache_ratio) # TODO relax this assumption
        self.sa_head = Head(head_size)
        self.cache_head = Head(self.cache_dim, n_embd=n_embd + self.cache_dim) # this is different from paper
        self.ratio = nn.Parameter(torch.tensor(0.))
        self.w_r = nn.Linear(n_embd + self.cache_dim, self.cache_dim)
        self.w_u = nn.Linear(n_embd + self.cache_dim, self.cache_dim)
        self.w_c = nn.Linear(n_embd + self.cache_dim, self.cache_dim)
        self.gr_cache = torch.zeros(1, block_size, self.cache_dim).to(device)
        #self.register_buffer('gr_cache', torch.zeros(1, block_size, self.cache_dim).to(device))

    def forward(self, x):
        # x : (B,T,C)
        B,T,C = x.shape
        z = torch.zeros(B,block_size-T,C, requires_grad=True).to(device)
        xz = torch.cat((z, x), dim=1)
        # cache : (B,T,D=rC^2)
        if self.gr_cache.shape[0] != B:
            #self.register_buffer('gr_cache', torch.zeros(B, block_size, self.cache_dim).to(device))
            self.gr_cache = torch.zeros(B, block_size, self.cache_dim).to(device)

        sa = self.sa_head(xz) # (B,T,H)

        cache = self.gr_cache
        xt1 = torch.cat((xz,cache), dim=-1)
        gu = F.sigmoid(self.w_u(xt1))
        gr = F.sigmoid(self.w_r(xt1))
        xt2 = torch.cat((xz, gr*cache), dim=-1)
        new_cache = (1-gu)*cache + gu*self.w_c(xt2)
        self.gr_cache = new_cache.detach()

        ca = self.cache_head(torch.cat((xz, new_cache), dim=-1))
        sr = F.sigmoid(self.ratio)
        return (sr*ca + (1-sr)*sa)[:, :T, :]

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size, head_fn=Head):
        super().__init__()
        self.heads = nn.ModuleList([head_fn(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHead(n_head, head_size)#, head_fn=GRCHead)
        #self.sa = grc.GRC_Self_Attention(n_embd, cache_ratio=cache_ratio, num_heads=n_head, dropout=dropout, device=device, spatial_pos_emb=False, batch_first=True)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, n_embd)
        self.pos_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        # self.sa_head = MultiHead(n_heads, n_embd//n_heads)
        # self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_table(idx) # (batch,time,channel)
        pos_emb = self.pos_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # x = self.sa_head(x)   # (B,T,C)
        # x = self.ffwd(x)      # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        B, T, C = logits.shape
        loss = None if targets is None else F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# -- training

model = GPT()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

stime = time.time()

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
        print()

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
print("training took", time.time() - stime, "seconds")



# -- RESULTS
# (these are all byte-level models)

# TRANSFORMER (~11M parameters, 3.8GB VRAM):
# step 0: train loss 4.2846, val loss 4.2820
# step 250: train loss 2.3222, val loss 2.3504
# step 500: train loss 1.8757, val loss 1.9932
# step 750: train loss 1.6577, val loss 1.8179
# step 1000: train loss 1.5308, val loss 1.7217
# step 1250: train loss 1.4511, val loss 1.6538
# step 1500: train loss 1.3888, val loss 1.6041
# step 1750: train loss 1.3437, val loss 1.5617
# step 2000: train loss 1.3087, val loss 1.5522
# step 2250: train loss 1.2785, val loss 1.5278
# step 3000: train loss 1.2033, val loss 1.4947
# step 4000: train loss 1.1220, val loss 1.4782
# step 4500: train loss 1.0886, val loss 1.4769
# step 5000: train loss 1.0530, val loss 1.4920
# step 6000: train loss 0.9827, val loss 1.5083
# step 7000: train loss 0.9104, val loss 1.5319
# step 7450: train loss 0.8790, val loss 1.5529
#
# training took 9060.934713125229 seconds
# (I was in power saver mode for the first ~800 iterations though)
