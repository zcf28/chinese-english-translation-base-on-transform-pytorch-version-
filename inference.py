import numpy as np
import torch
from torch.autograd import Variable
from dataset import Dataset
from model.make_model import make_model

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


def load_model(save_model_path, model):
    model_pth = torch.load(save_model_path)
    model.load_state_dict(model_pth)
    return model.eval()


def evaluate(data, model):
    """
    在data上用训练好的模型进行预测，打印模型翻译结果
    """

    MAX_LENGTH = 60  # 语句最大长度

    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(len(data.dev_en)):
            # 打印待翻译的英文语句
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)

            # 打印对应的中文语句答案
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))

            # 将当前以单词id表示的英文语句数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH,
                                start_symbol=data.cn_word_dict["BOS"])
            # 初始化一个用于存放模型翻译结果语句单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文语句结果
            print("translation: %s" % " ".join(translation))


if __name__ == '__main__':
    train_file_path = f"./datasets/en-cn/train_mini.txt"
    dev_file_path = f"./datasets/en-cn/dev_mini.txt"
    batch_size = 1

    data = Dataset(train_file_path, dev_file_path, batch_size, DEVICE)

    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)

    # 初始化模型
    layer_nums = 6  # transformer中encoder、decoder层数
    d_model = 512  # 输入、输出词向量维数
    d_ff = 1024  # feed forward全连接层维数
    h_num = 8  # 多头注意力个数

    model = make_model(src_vocab, tgt_vocab, layer_nums, d_model, d_ff, h_num, device=DEVICE)

    model = load_model(f"./save_model/epoch_200.pth", model).to(DEVICE)

    evaluate(data, model)
