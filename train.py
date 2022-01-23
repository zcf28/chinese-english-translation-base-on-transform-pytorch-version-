import logging
import time

import torch
from tqdm import tqdm

from dataset import Dataset
from model.make_model import make_model
from model.label_smoothing import LabelSmoothing
from model.noam_opt import NoamOpt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEVICE = "cpu"


def train(epochs, batch_size, lr):
    logging.basicConfig(filename="./run.log", level=logging.INFO, filemode="w")

    # 数据预处理
    train_file_path = f"./datasets/en-cn/train_mini.txt"
    dev_file_path = f"./datasets/en-cn/dev_mini.txt"

    data = Dataset(train_file_path, dev_file_path, batch_size, DEVICE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % src_vocab)
    print("tgt_vocab %d" % tgt_vocab)

    # 初始化模型
    layer_nums = 6  # transformer中encoder、decoder层数
    d_model = 512  # 输入、输出词向量维数
    d_ff = 1024  # feed forward全连接层维数
    h_num = 8  # 多头注意力个数
    drop_out = 0.1  # dropout比例

    model = make_model(src_vocab, tgt_vocab, layer_nums, d_model, d_ff, h_num, drop_out, DEVICE)
    model.to(DEVICE)
    logging.info(f"model arch:\n {model}")

    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.0)
    optimizer = NoamOpt(d_model, 1, 2000, torch.optim.Adam(model.parameters(),
                                                           lr=lr, betas=(0.9, 0.98), eps=1e-9))

    for epoch in tqdm(range(epochs)):
        model.train()

        total_tokens = 0.0
        total_loss = 0.0
        tokens = 0.0

        for i, batch in enumerate(data.train_data):
            optimizer.optimizer.zero_grad()

            model_out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            gen_out = model.generator(model_out)

            loss = criterion(gen_out.contiguous().view(-1, gen_out.size(-1)),
                             batch.trg_y.contiguous().view(-1)) / batch.ntokens

            loss.backward()
            optimizer.step()

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

        train_loss = total_loss / total_tokens

        logging.info(f"epoch: {epoch}, train_loss: {train_loss:0.8f}")

        if epoch % 100 == 0 and epoch != 0:
            # dev_total_tokens = 0.0
            # dev_total_loss = 0.0
            # dev_tokens = 0.0
            #
            # model.eval()
            # for i, dev_batch in enumerate(data.dev_data):
            #     dev_model_out = model(dev_batch.src, dev_batch.trg, dev_batch.src_mask, dev_batch.trg_mask)
            #     dev_gen_out = model.generator(dev_model_out)
            #     dev_loss = criterion(dev_gen_out.contiguous().view(-1, dev_gen_out.size(-1)),
            #                          dev_batch.trg_y.contiguous().view(-1)) / dev_batch.ntokens
            #
            #     dev_total_loss += dev_loss
            #     dev_total_tokens += dev_batch.ntokens
            #     dev_tokens += dev_batch.ntokens
            #
            # dev_loss = dev_total_loss / dev_total_tokens
            #
            # logging.info(f"= =" * 10)
            #
            # logging.info(f"epoch: {epoch}, train_loss: {dev_loss:0.8f}")
            #
            # logging.info(f"= =" * 10)

            logging.info(f"= =" * 5 + "SAVE MODEL" + "= = " * 5)
            save_model_path = f"./save_model/epoch_{epoch}.pth"
            torch.save(model.state_dict(), save_model_path)


if __name__ == '__main__':
    epochs = 1000
    batch_size = 32
    lr = 1e-4

    train(epochs, batch_size, lr)
