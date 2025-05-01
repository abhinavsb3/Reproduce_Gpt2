from dataclasses import dataclass
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time
import inspect
import numpy as np
import sys
from hellaswag import render_example,iterate_examples, get_most_likely_row
#____________________________________________________________________________


class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B,nh,T,T) x (B, nh, T, hs) -> (B,nh,T,hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) #flash attention

        y = y.transpose(1,2).contiguous().view(B, T, C) # (B, T, C) basically the concat operation of differnt heads
        y = self.c_proj(y)
        return y
    


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 #number of tokens: 50000 BPE merges + 256 byte tokens +1 special token which is endoftext
    n_layer: int = 12 #number of layers
    n_head: int = 12 #number of heads
    n_embd: int = 768 #embedding dimensions

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <=self.config.block_size, f"Cannot forward sequence of length {T} ,block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" %model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] =1024

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
         
        #for build hugging face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #  just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] #  just the mask (buffer)
        #we need to transpose some weight matrix since it strored in (out,in) format in hf model because its using conv1D.
        #so after transpose we will get those weight matrixes as (input,output) which is in pytorch format.
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
       #taking all candidate parameters that require grad
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        #creating Optim groups that any parameters that 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [{'params':decay_params, ' weight_decay': weight_decay},
                       {'params':nodecay_params, 'weight_decay': 0.0}
                       ]
        num_decay_params = sum(p.numel() for p in decay_params)  
        num_nodecay_params = sum(p.numel() for p in nodecay_params) 
        if master_process:
            print(f"num decayed parameters tensors: {len(decay_params)}, with{num_decay_params}:parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"    #Kernal fusion for optimizer calculations
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
        return optimizer   

#_______________________________________________________________________________

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

#Data loader
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        #get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards)> 0, f"no shards found for split {split}"   
        if master_process:
            print(f"found {len(shards)} shards for split {split}")  
        self.reset() 

    def reset(self):
    #state, init at shard 0
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank 

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T+1]
        x = (buf[:-1]).view(B,T) #input
        y = (buf[1:]).view(B,T) #targets
        
        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y



    #_________________________________________________________

#Setting up DDP
#torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
#run training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1 #will be True if ddp run
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 #this is the process doing checkpoint,logging,etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #attempt to autodetect the device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  #for mac users use apple silicon cpu which allready have gpu.mps is backend for apple silicon
    print(f"Using device: {device}")
# device = "cpu" #OVERRIDE

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens.Batch size in gpt2 paper = 524288
#384/524288
B = 64 #4/16/64 #micro batch size
T = 1024 #32/1024 #sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0 #confirmimg total_batch_size is divisible by B * T * ddp_worldsize
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train") #(4,32)/(16,1024)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high') #set fp32 precision.Set hifh so everything will be in tensor float 32(tf32)

#Create Model
model = GPT(GPTConfig(vocab_size=50304)) #Changed vocab_size fro 50257 to 50304 for optimization and efficencysince it is a power of 2
model.to(device)
use_compile = True #ON and OFF point of torch.compile
if use_compile:
    model = torch.compile(model) 
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 #we are doing 524288 tokens per step and we have 10B token.so 10B/524288 = 19073 
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#optimzer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)


#creating the log directory.Will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    #once in a while evaluate validation loss
    if step % 350 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 or last_step: #save checkpoint in a every validation
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'config': raw_model.config
                    #'torch_rng_state': torch.get_rng_state(),
                    #'cuda_rng_state': torch.cuda.get_rng_state_all(),
                    #'numpy_rng_state': np.random.get_state()                   
                }
                torch.save(checkpoint, checkpoint_path)


    #Evaluating Hellaswag once in a while
    if (step % 250 == 0 or last_step) and (not use_compile): 
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process example where i % ddp_world_size ==ddp_rank#this is for proper managemnt of which part is deal by which gpu
            if ddp:
                if i % ddp_world_size != ddp_rank:
                    continue
            #rendering example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            #get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        #reduce the stats accross all process
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total #accuracy of hellaswag
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")        


    # #Generate once in a while
    # if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile) :
    #     model.eval()
    #     num_return_sequences = 4
    #     max_length = 32
    #     enc = tiktoken.get_encoding('gpt2')
    #     tokens = enc.encode("Hello, I'm a language model,")
    #     tokens = torch.tensor(tokens, dtype=torch.long)
    #     tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    #     xgen = tokens.to(device)
    #     sample_rng = torch.Generator(device=device)
    #     sample_rng.manual_seed(42 + ddp_rank)
    #     while xgen.size(1) < max_length:
    #         with torch.no_grad():
    #             logits, loss = model(xgen) #(B, T, vocab_size)
    #             logits = logits[:, -1, :] #(B, vocab_size)
    #             probs = F.softmax(logits, dim=-1) #get probabilities
    #             topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) #topk sampling for top 50 probabilities
    #             ix = torch.multinomial(topk_probs, 1, generator=sample_rng)#(B,1),selecting a token from topk 
    #             xcol = torch.gather(topk_indices, -1, ix)#gathering corresponding indices
    #             xgen = torch.cat((xgen,xcol), dim = 1)#append to sequence
    #     #print generated sequence
    #     for i in range(num_return_sequences):
    #         tokens = xgen[i, :max_length].tolist()
    #         decoded = enc.decode(tokens)
    #         print(f"rank {ddp_rank} sample {i}: {decoded}")         


    #training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) #this line will mkae ddp to synchronize gpus only for last loop microstep and sync off in all other steps
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): #using mixed precision
            logits, loss = model(x, y)
        #Watch video for understand need of scalling in loss calculated below
        loss = loss / grad_accum_steps #loss scaling.otherwise weight gradient wont we same as normal loss without microstep.reason explained in video
        loss_accum += loss.detach() #used detach for not include this in computational graph
                                    #since loss is scaled loss which is divided by grad_accum_step it will be a small value so the cummulated loss will be real loss to print.
        loss.backward()
    
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) #loss accum will be the average of loss accum in all gpus

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient clipping
    #determine and set learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == 'cuda':
        torch.cuda.synchronize() #it will make a que for next process till completing current process in gpu
    t1 = time.time()
    dt = (t1-t0)*1000 #time difference in milliseconds#remove *1000 for time in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps  * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step:{step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} |  norm:{norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | " )
        with open(log_file, 'a') as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
if ddp:
    destroy_process_group()
#sys.exit(0)