import copy

import torch
from torch import nn

from model.multi_headed_attention import MultiHeadedAttention
from model.positionwise_feed_forward import PositionwiseFeedForward
from model.positional_encoding import PositionalEncoding
from model.encoder import EncoderLayer, Encoder
from model.decoder import DecoderLayer, Decoder
from model.transform import Transformer
from model.embeddings import Embeddings
from model.generator import Generator


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, device=None):

    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model).to(device)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(device)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout, device).to(device)
    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(device), N).to(device),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(device), N).to(device),
        nn.Sequential(Embeddings(d_model, src_vocab).to(device), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(device), c(position)),
        Generator(d_model, tgt_vocab)).to(device)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(device)