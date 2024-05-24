from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import time

@dataclass
class ModelConfig:
    """
    Design your Memory Mosaic Model here
    Yes I know dropout_rate should probably be in TrainConfig but it was easier to implement from here
    """
    # general hyperparameters
    dim: int = 128
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu' 
    dropout_rate = 0.1 # percent of neurons to set to 0 during training as a way of adding randomness & improving generalization

    # tokenizer
    tokenizer: str = 'bpe_v1' # must choose from one of the folders in 'tokenizers/'. current options: 'bpe_v1', 'bpe_v2'
    vocab_len: int = 8192 # options assuming 'bpe' are 95 (character-wise), 128, 256, 512, 1024, 2048, 4096, & 8192
    # ^ that number does not include the three tokens bos, eos, and pad

    # Residual Layers
    num_layers: int = 4 # small models should err on the side of many many layers at the expense of attention & mlp sizes
    second_resid_norm: bool = False # True adds an extra Norm after the attn & MLP, like in Grok. Only recommended if using RMSNorm

    # Memory Mosaic (mm)
    num_heads: int = 4
    head_dim: int = dim // num_heads # the mm authors only tried dim // num_heads
    max_seq_len: int = 128 # 512 is the most my 8gb of ram can handle
    mm_bias: bool = False # whether to use bias in mm matrices
    pmem_size: int = 2688 // (1024 // dim) # this calculation keeps the same ratio as what they used in their model
    pmem_count: int = 2
    # might I split these hyperparameters up to be separate for context vs persistent memory?
    # also could mix & match gpt parts with mm parts and see how that goes

    # normalization
    scale_first_resid: bool = True # whether to multiply the first residual state by sqrt(dim)
    norm_type: str = 'RMSNorm' # options are 'RMSNorm'(recommended), 'LayerNorm', and 'CosineNorm'.
    norm_affine: bool = True # whether to use a linear layer after each norm. recommended especially if you're using LayerNorm or CosineNorm
    norm_bias: bool = True # whether to add a bias to the linear layer after each norm. doesn't do anything if norm_affine == False
    eps: float = 1e-6 # small constant to prevent division by 0. Not really worth editing

@dataclass
class TrainConfig:
    """
    Design your training loop here
    """
    # name of the folder the model will be saved into
    model_name = f'{time.strftime("%Y-%m-%d|%H-%M-%S")}' # defaults to the time that config.py was imported
    
    weight_decay: float = 0.05
    batch_size: int = 32
    
    # total number of batches to run over the course of training
    max_iters: int = 100 # i recommend at least 1_000
    # how often to print out an update on how training is going
    eval_interval: int = max_iters // 100 # doing this too often slows things down hella but also gives detailed log data
    # how many samples to take at each evaluation. more means a more accurate loss/perplexity calculation
    eval_samples: int = 1 # this number can slow things down. each sample is almost like doing an extra training iteration
    # how often to save a model checkpoint
    checkpoint_interval: int = None # eval_interval # set to None if you don't want checkpoints
    
    ### to visualize the learning rate schedule you define here, see cell 7 of training.ipynb

    # Initial learning rate to start from during the warmup
    lr_init: float = 1e-6
    # Maximum and minimum learning rates during annealing
    lr_max: float = 1e-1
    lr_min: float = 1e-3
    # if you'd like a flat learning rate, set lr_init = lr_min = lr_max and ignore the variables below
    
    # number of iterations for a linear warmup from lr_min to lr_max
    warmup_iters: int = int(max_iters * 0.01) # if you don't want to use a lr warmup, set = 0
    # number of iterations for a constant learning rate of lr_min at the end of training
    final_flat_iters: int = int(max_iters * 0.1) # if you don't want to use a final flat lr at the end, set = 0
    
    # type of annealment to use. Annealment is when the learning rate decreases over the course of training
    anneal_type: str = 'cos' # options: 'cos'(recommended) and 'lin'
    # number of times to bring the learning rate back up from lr_min to lr_max in-between the warmup & final flat
    num_restarts: int = 3 # if you don't want to use warm restarts, set =0 and ignore T_mult
    # relative length of each warm restart compared to the previous.
    T_mult: int = 2 # =1 makes all restarts the same length, <1 means they get shorter and >1 makes them longer
    
    # Calculates T_0 in a way that ensures smooth transition to the final flat learning rate
    def T_0(self): # I DO NOT RECOMMEND EDITING THIS
        middle_section = self.max_iters - self.warmup_iters - self.final_flat_iters
        return middle_section / sum(self.T_mult ** i for i in range(self.num_restarts+1))