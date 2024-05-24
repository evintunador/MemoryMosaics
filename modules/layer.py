import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from modules.logging import LoggingModule, log_io
from modules.norm import Norm
#from modules.mqa import MQA
#from modules.mlp import MLP
from modules.memory_mosaic import ContextMem, PersistentMem

class Layer(LoggingModule):
    def __init__(self, cfg):
        super().__init__()
        self.second_norm = cfg.second_resid_norm
        self.dropout_rate = cfg.dropout_rate

        # context memories connection (analogous to the attention mechanism)
        self.pre_context_norm = Norm(
            cfg.dim, 
            cfg.norm_type, 
            cfg.norm_affine, 
            cfg.norm_bias, 
            cfg.eps
        )
        self.context = ContextMem(
            cfg.num_heads, 
            cfg.head_dim,
            cfg.dim, 
            cfg.mm_bias, 
            cfg.max_seq_len, 
            cfg.dropout_rate
        )
        if self.second_norm: 
            self.post_context_norm = Norm(
                cfg.dim, 
                cfg.norm_type, 
                cfg.norm_affine, 
                cfg.norm_bias, 
                cfg.eps
            )

        # persistent memories connection (analogous to the feedforward network)
        self.pre_persistent_norm = Norm(
            cfg.dim, 
            cfg.norm_type, 
            cfg.norm_affine, 
            cfg.norm_bias, 
            cfg.eps
        ) 
        self.persistent = PersistentMem(
            cfg.num_heads, 
            cfg.head_dim,
            cfg.dim, 
            cfg.mm_bias, 
            cfg.max_seq_len, 
            cfg.pmem_count, 
            cfg.pmem_size, 
            cfg.dropout_rate
        )
        if self.second_norm: 
            self.post_persistent_norm = Norm(
                cfg.dim, 
                cfg.norm_type, 
                cfg.norm_affine, 
                cfg.norm_bias, 
                cfg.eps
            )

    @log_io
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = x + self.context_connect(x, training)
        x = x + self.persistent_connect(x, training)
        return x

    @log_io
    def context_connect(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        dx = self.context(self.pre_context_norm(x), training)
        if self.second_norm: dx = self.post_context_norm(dx)
        return dx

    @log_io
    def persistent_connect(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        dx = self.persistent(self.pre_persistent_norm(x), training)
        if self.second_norm: dx = self.post_persistent_norm(dx)
        return dx