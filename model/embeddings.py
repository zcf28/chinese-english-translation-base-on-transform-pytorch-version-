import math

import torch
from torch import nn


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x的词向量（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


if __name__ == '__main__':

    vocab = 100  # 嵌入层字典的大小（单词本里单词个数）
    d_model = 512  # 每个产出向量的大小
    embed = Embeddings(d_model, vocab)

    x_data = torch.randint(0, 10, (16, 10))
    print(x_data, x_data.shape)

    out_data = embed(x_data)
    print(out_data, out_data.shape)
