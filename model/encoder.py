import copy

import torch
from torch import nn

from model.sublayer_connection import SublayerConnection
from model.utils import clones
from model.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection作用连接multi和ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # attn的结果直接作为下一层输入
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        layer = EncoderLayer
        """
        super(Encoder, self).__init__()
        # 复制N个编码器基本单元
        self.layers = clones(layer, N)
        # 层归一化
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        循环编码器基本单元N次
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


if __name__ == '__main__':

    from multi_headed_attention import MultiHeadedAttention
    from positionwise_feed_forward import PositionwiseFeedForward

    size = 512
    self_attn = MultiHeadedAttention(h=16, d_model=512, dropout=0.5)
    feed_forward = PositionwiseFeedForward(d_model=512, d_ff=512, dropout=0.5)
    dropout = 0.5

    c = copy.deepcopy

    encoder_layer = EncoderLayer(size, c(self_attn), c(feed_forward), dropout)

    x_data = torch.rand(16, 10, 512)

    # out_data = encoder_layer(x_data, mask=None)
    #
    # print(out_data, out_data.shape)

    encoder = Encoder(encoder_layer, 8)
    print(encoder)

    out_data = encoder(x_data, mask=None)
    print(out_data, out_data.shape)