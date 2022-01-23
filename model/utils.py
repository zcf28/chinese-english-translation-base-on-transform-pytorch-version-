import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


def clones(module, N):
    """
    克隆基本单元，克隆的单元之间参数不共享
    """
    return nn.ModuleList([
        copy.deepcopy(module) for _ in range(N)
    ])


def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention（方程（4））
    """
    # q、k、v向量长度为d_k
    d_k = query.size(-1)
    # 矩阵乘法实现q、k点积注意力，sqrt(d_k)归一化
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 注意力掩码机制
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 注意力矩阵softmax归一化
    p_attn = F.softmax(scores, dim=-1)
    # dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 注意力对v加权
    return torch.matmul(p_attn, value), p_attn