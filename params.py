import random
from typing import Literal
import torch
import os
import numpy as np
from tap import Tap


class Params(Tap):
    ckpt: str = 'no.ckpt'  # 检查点
    save_dir: str = 'model_out'  # 模型保存输出
    label_fn: str = 'union_label.txt'  # 标签
    slot_label: str = 'slot_label.txt'  # 槽填充标签
    #bert: str = 'bert-base-chinese'  # BERT输出
    bert: str = 'bert-base-uncased'  # BERT输出
    dataset: str = 'SimM'  # 数据集路径 custom 
    fine_tune_steps: int = 100000
    dropout: float = 0.3
    hist_hidden: int = 256
    utter_hidden: int = 256
    gcn_layers: int = 2  # 图卷积层数
    gcn_out_dim = 256  # 图卷积输出维度
    gcn_hidden_dim = 256  # GCN隐藏层的维度
    slot_embed_dim: int = 32  # slot embedding的维度
    num_intent_embed: int = 10
    context_window: int = 1
    enable_context: bool = True  # 设定固定的历史对话轮次窗口 - 控制model的规模
    gpu: int = 0  # 设置GPU
    max_epoch: int = 100 #########################################################################################################
    eval_steps: int = 5000
    save_steps: int = 5000
    warmup_steps: int = 15000
    lr: float = 2e-5
    accumulated_steps = 1
    alpha_1: float = 1000.0
    alpha_2: float = 3500.0
    slot_mlp: bool = False
    stack: bool = True  # --stack
    slow_lr = 2e-6
    fast_lr = 2e-4
    # MARK: Stack SLU
    word_embedding_dim: int = 128
    slot_decoder_hidden_dim: int = 128
    intent_decoder_hidden_dim: int = 128
    attention_hidden_dim: int = 1024
    attention_output_dim: int = 128
    encoder_hidden_dim: int = 256
    differentiable: bool = False
    seed: int = 7890

    @property
    def device(self):
        """if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")"""
        if self.gpu >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{self.gpu}")
        else:
            return torch.device('cpu')

    def seed_anything(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
