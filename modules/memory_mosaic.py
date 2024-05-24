#
# This file is derived from the Memory Mosaics Repo. changes are fairly minimal, mostly chainging variable names to fit my preferences
# https://github.com/facebookresearch/MemoryMosaics/blob/main/nanoMosaics/mosaic_model.py
# 

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.logging import LoggingModule, log_io

class LeakyAvg(LoggingModule):
    def __init__(self, max_seq_len, num_heads):
        super().__init__()
        coef = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            coef = torch.diagonal_scatter(coef, -torch.ones(max_seq_len-i)*i, -i)
        self.register_buffer('coef', coef)
        self.exp_scaling = 10
        self.leaky_key_beta = nn.Parameter(torch.linspace(0.5, 5, num_heads).view(1, num_heads, 1, 1)/self.exp_scaling)

    @log_io
    def forward(self, k): 
        batch_size, num_heads, seq_len, head_dim = k.size()
        leaky_key_beta = self.leaky_key_beta.abs() * self.exp_scaling
        coef = self.coef[:seq_len,:seq_len].view(1,1,seq_len,seq_len)
        coef = torch.exp(coef * leaky_key_beta)
        return coef.tril() @ k

class KeyFeatureExtractor(LoggingModule):
    def __init__(self, num_heads, head_dim, dim, mm_bias, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W_k = nn.Linear(dim, num_heads * head_dim, bias=mm_bias)
        self.leaky_avg = LeakyAvg(max_seq_len, num_heads)
        self.exp_scaling = 10
        self.key_scale = nn.Parameter(torch.ones(1, num_heads, 1, 1) / self.exp_scaling)
        self.key_scale_max = math.log(2**16-1) # fits in fp16.

    @log_io
    def forward(self, x, scale_pow=1):
        k = self.make_key(x)
        k = self.leaky_avg(k)
        k = self.scale_key(k, scale_pow)
        return k

    @log_io
    def make_key(self, x):
        batch_size, seq_len, dim = x.size()
        k = self.W_k(x).transpose(1,2).view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(2,3)
        return k

    @log_io
    def scale_key(self, k, scale_pow):
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-10)
        k = k * (scale_pow * self.exp_scaling * self.key_scale).clamp(max=self.key_scale_max).exp()
        return k

class ValFeatureExtractor(LoggingModule):
    def __init__(self, num_heads, head_dim, dim, mm_bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        v_shift = 1 # access to x_T+1
        self.shift_fn = lambda x: F.pad(x, (-v_shift, v_shift))
        self.W_v = nn.Linear(dim, num_heads * head_dim, bias=mm_bias)
        self.coef = nn.Parameter(torch.rand(1, num_heads, 1, 1))
        self.exp_scaling = 10
        val_scale_init = -.5
        self.val_scale  = nn.Parameter(torch.ones(1, num_heads, 1, 1) * val_scale_init / self.exp_scaling)

    @log_io
    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        v = self.make_val(x)
        v = self.scale_val(v)
        return v

    @log_io
    def make_val(self, x):
        batch_size, seq_len, dim = x.size()
        v = self.W_v(x).transpose(1,2).view(batch_size, self.num_heads, self.head_dim, seq_len)
        return v

    @log_io
    def scale_val(self, v):
        v = (1-self.coef) * self.shift_fn(v) + self.coef * v
        v = v.transpose(2,3)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-10)
        v = v * (self.exp_scaling * self.val_scale).exp()
        return v

class ContextMem(LoggingModule):
    def __init__(self, num_heads, head_dim, dim, mm_bias, max_seq_len, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        assert dim % num_heads == 0
        # key, value projections for all heads, but in a batch
        self.k_featurizer = KeyFeatureExtractor(num_heads, head_dim, dim, mm_bias, max_seq_len)
        self.v_featurizer = ValFeatureExtractor(num_heads, head_dim, dim, mm_bias)

        # output projection
        self.c_proj = nn.Linear(num_heads * head_dim, dim, bias=mm_bias)
        #self.c_proj = nn.Linear(dim, dim, bias=mm_bias)
        
        # regularization
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.dropout = dropout_rate
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len), diagonal=-1)
                                    .view(1, 1, max_seq_len, max_seq_len).bool())

    @log_io
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()

        # calculate key, values for all heads in batch and move head forward to be the batch dim
        k = self.k_featurizer(x) # batch_size, num_heads, seq_len, head_dim
        v = self.v_featurizer(x) # batch_size, num_heads, seq_len, head_dim

        # causal self-attention; Self-attend: 
        # (batch_size, num_heads, seq_len, head_dim) x (batch_size, num_heads, head_dim, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        att = self.attend(k, seq_len, training)
        
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        y = self.proj_vals(att, v)

        y = self.reassemble_heads(y, batch_size, seq_len)

        # output projection
        y = self.c_proj(y)
        return self.resid_dropout(y) if training else y

    @log_io
    def attend(self, k, seq_len, training):
        att = (k[:,:,1:] @ k.transpose(-2, -1))
        att = att.masked_fill(~self.bias[:,:,1:seq_len,:seq_len], float('-inf'))
        att = F.softmax(att, dim=-1)
        if training: att = self.attn_dropout(att)
        return att

    @log_io
    def proj_vals(self, att, v):
        y = torch.zeros_like(v)
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        y[:, :, 1:] = att @ v 
        return y

    @log_io
    def reassemble_heads(self, y, batch_size, seq_len):
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return y

class PersistentMem(LoggingModule):
    def __init__(self, num_heads, head_dim, dim, mm_bias, max_seq_len, pmem_count, pmem_size, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # key, value projections for all heads, but in a batch
        self.k_featurizer = KeyFeatureExtractor(num_heads, head_dim, dim, mm_bias, max_seq_len)
        self.P_k = nn.Parameter(torch.zeros(pmem_count, 1, num_heads, pmem_size, head_dim))
        self.P_v = nn.Parameter(torch.zeros(pmem_count, 1, num_heads, pmem_size, head_dim))
        self.exp_scaling = 10
        out_scale_init = -.5
        self.out_scale  = nn.Parameter(torch.ones(1, num_heads, 1, 1) * out_scale_init / self.exp_scaling)
        torch.nn.init.normal_(self.P_k, mean=0.0, std=1 / math.sqrt(head_dim))
        torch.nn.init.normal_(self.P_v, mean=0.0, std=1 / math.sqrt(head_dim))

        # output projection
        self.c_proj = nn.Linear(num_heads * head_dim, dim, bias=mm_bias)
        
        # regularization
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.dropout = dropout_rate
        self.pmem_count = pmem_count
        self.attn_dropout = nn.Dropout(dropout_rate)

    @log_io
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()

        # calculate key, values for all heads in batch and move head forward to be the batch dim
        k = self.k_featurizer(x, scale_pow=2) # 2 because P_k does not have scale

        y = 0
        for i in range(self.pmem_count):
            att = self.attend(y, k, i)
            y += self.proj_val(att, i)
            
        y = self.scale(y)
        y = self.reassemble_heads(y, batch_size, seq_len)

        # output projection
        y = self.c_proj(y)
        return self.resid_dropout(y) if training else y

    @log_io
    def attend(self, y, k, i):
        att = k @ (self.P_k[i].transpose(-2, -1))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att

    @log_io
    def proj_val(self, att, i):
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        return att @ self.P_v[i]

    @log_io
    def scale(self, y):
        y = y / self.pmem_count
        y = y * (self.exp_scaling * self.out_scale).exp()
        return y

    @log_io
    def reassemble_heads(self, y, batch_size, seq_len):
        return y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

# these are the original hyperparameters from https://github.com/facebookresearch/MemoryMosaics/blob/main/nanoMosaics/mosaic_model.py
#@dataclass
#class MemoryMosaiccfg:
#    max_seq_len: int = 1024
#    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
#    n_layer: int = 12
#    num_heads: int = 12
#    dim: int = 768
#    dropout: float = 0.0
#    pmem_size: int = 2688
#    pmem_count: int = 1
#    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
#    leaky_cuda: bool = False # True: use LeakyAverageCuda, False: use LeakyAvg

