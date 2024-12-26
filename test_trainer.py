import litdata.utilities
import litdata.utilities.env
from tiny_trainer import *
import numpy as np
import torch
import hydra
import torch.nn as nn
from torch.utils.data import *
from typing import *
# GPT2 model and config
from model import *
# force the same seed
set_all_seed(114514)
from litdata import *
import litdata

from dataclasses import dataclass

def create_scheduler(optimizer, warmup_step, total_step):
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    
    # Create schedulers
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1e-7,  # Start from lr = 0
        end_factor=1.0,    # End with the base lr
        total_iters=warmup_step    # Warmup steps
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_step - warmup_step,  # Remaining steps after warmup
        eta_min=0        # Minimum learning rate
    )
    
    # Chain them together
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_step]    # Switch to cosine scheduler after warmup
    )
    return scheduler

# optimizer = torch.optim.Adam([torch.tensor(0)], lr=1)
# scheduler = create_scheduler(optimizer=optimizer, warmup_step=700, total_step=20000)
# lrs = []
# for step in range(20000):
#     # Training code here
#     optimizer.step()
#     scheduler.step()
#     lr = scheduler.get_last_lr()[-1]
#     lrs.append(lr)
# import matplotlib.pyplot as plt
# plt.plot(lrs)

@dataclass
class AllConfig:
    model : str = "d12"
    learning_rate : float = 6e-4
    weight_decay : float = 0.1
    warmup : int = 700
    total_step = 19000
    data_dir : int = "/home/zzhang18/nchen3/test-10b/"
    data_seed : int = 42
    seq_len : int = 1024
    # batch size measured with sequences
    # total batch size = 512
    batch_size : int = 16
    # False == llm.c style
    inter_shuffle : bool = False
    outer_shuffle : bool = False

print0 = print
args = AllConfig()

class SafeStreamingDataset(StreamingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        # note : the dataset must be created after pytorch has initialized the process group
        # after the dataloader can't see the rank
        
        env = litdata.utilities.env._DistributedEnv.detect()
        print("Detected env:", env)
        assert env.global_rank == torch.distributed.get_rank()
        print("Succesfully create dataset after initialization")
    def __getitem__(self, index):
        t = super().__getitem__(index)
        return t.copy()

class TestFactory(AbstractTrainerFactory):
    def make_model(self) -> nn.Module:
        model_config = {
            "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
            "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
            "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
            "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
        }[args.model]
        model = GPT(model_config)
        return model
    def make_optimizer(self, model) -> torch.optim.Optimizer:
        optimizer = configure_optimizers(model, weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type="cuda", zero_stage=1)
        return optimizer
    def make_dataloader(self, rank : int) -> Tuple[Any, DataLoader, Optional[DistributedSampler]]:     
        if args.inter_shuffle:
            batch_size=args.batch_size
            item_length = args.seq_len + 1
        else:
            batch_size=1
            item_length = args.batch_size*args.seq_len+1
        dataset = SafeStreamingDataset(
            input_dir=args.data_dir,
            item_loader=TokensLoader(item_length),
            seed = args.data_seed,
            shuffle=args.outer_shuffle,
            drop_last=True,
        )
        train_dataloader = StreamingDataLoader(dataset, batch_size=batch_size, pin_memory=True, drop_last=True, shuffle=args.outer_shuffle, num_workers=4)
        # rank and world_size is fetched automatically
        # seed must be the same across the cluster
        # sampler = torch.utils.data.DistributedSampler(dataset=dataset, seed=0)
        print("dataloader length", len(train_dataloader))
        print("dataset length", len(dataset))
        sampler = None
        return (None, train_dataloader, sampler)
    def make_scheduler(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        # no change...
        scheduler = create_scheduler(optimizer, warmup_step=args.warmup, total_step=args.total_step)
        return scheduler

class TestTrainer(DefaultTrainer):
    def on_training_begin(self):
        super().on_training_begin()
        if self.rank == 0:
            print(self.scheduler)
            print(self.optimizer)
            print(self.model)
    def val_model(self, i, is_debug=False):
        pass
    def eval_model(self, i, is_debug=False):
        pass
    def prepare_input(self, obj):
        return {"x" : obj.to(device=self.device, dtype=torch.long)}
    def calculate_loss(self, x):
        # we process data differently here
        # llm.c generate a 1D array, so no need for contiguous
        # but we produce a (B, seq_len + 1) batch
        # a lot of copy here, maybe change to better representation?
        # import tiktoken
        # enc = tiktoken.get_encoding("gpt2")
        # print(enc.decode(x[0].tolist()))
        # print(x.shape)
        if args.inter_shuffle:
            i = x[:, :-1].contiguous().detach()
            o = x[:, 1:].contiguous().detach()
        else:
            i = x.view(-1)[:-1].view(args.batch_size, args.seq_len).detach()
            o = x.view(-1)[1:].view(args.batch_size, args.seq_len).detach()
        _, loss = self.model(i, o, return_logits=False)
        return loss