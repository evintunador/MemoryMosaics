import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.logging import LoggingModule, log_io
from modules.norm import Norm
from modules.layer import Layer

class Model(LoggingModule):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.vocab_len = cfg.vocab_len + 3 # the 3 is the bos, eos, and padding tokens
        
        self.token_embedder = nn.Embedding(self.vocab_len, cfg.dim, device=cfg.device)
        self.scale = cfg.dim ** 0.5 if cfg.scale_first_resid else 1.0
        
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_layers))
        self.final_norm = Norm(
            cfg.dim, 
            cfg.norm_type, 
            cfg.norm_affine, 
            cfg.norm_bias, 
            cfg.eps
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index = self.vocab_len - 1) # ignore the padding token

    @log_io
    def forward(
        self, 
        input_token_ids: torch.Tensor, 
        target_token_ids: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor):
        input_token_ids = input_token_ids.to(self.device)
        if target_token_ids is not None:
            target_token_ids = target_token_ids.to(self.device)
        
        batch_size, seq_len = input_token_ids.shape
        
        if target_token_ids is not None: # if training
            assert input_token_ids.shape == target_token_ids.shape
            assert seq_len == self.max_seq_len
            training = True
        else: # if performing inference
            training = False
        
        # initialize first residual state and run the model
        x = self.token_embedder(input_token_ids) * self.scale # [batch_size, seq_len, dim]
        for layer in self.layers:
            x = layer(
                x, 
                training,
            )
        x = self.final_norm(x)
        logits = x @ self.token_embedder.weight.t() # [batch_size, seq_len, vocab_len]

        if training:
            loss = self.criterion(
                logits.view(batch_size * seq_len, self.vocab_len),
                target_token_ids.reshape(batch_size * seq_len)
            )
        else:
            loss = None
            
        return logits, loss