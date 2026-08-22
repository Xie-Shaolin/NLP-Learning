# ========== Seq2Seq 模型定义 ==========
# 经典"编码器-解码器"结构：
#   TranslationEncoder 把中文句子编码成一个语义向量（context vector）；
#   TranslationDecoder 以该向量为初始隐藏状态，逐个单词生成英文译文。
# 两者内部都是"Embedding(词嵌入) + GRU(循环神经网络) + 线性输出层"。
import torch
from torch import nn
import config


class TranslationEncoder(nn.Module):
    """编码器：把中文句子 -> 一个固定维度的语义向量（作为解码器的初始隐状态）。

    继承 nn.Module：所有 PyTorch 网络都继承它，从而获得参数注册、GPU 迁移、
    自动求导、state_dict 存取等基础设施能力。
    """

    def __init__(self, vocab_size, padding_index):
        """初始化编码器：创建 Embedding 层和 GRU 层。

        :param vocab_size: 源语言（中文）词表大小
        :param padding_index: <pad> 的索引，Embedding 用它屏蔽填充位
        """
        super().__init__()
        # nn.Embedding(num_embeddings, embedding_dim)：词嵌入查找表，
        # 输入整数索引 -> 输出对应词向量。padding_idx 指定的索引，其梯度恒为 0，padding 位不参与学习。
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

        # nn.GRU：门控循环单元。input_size 是输入维度（词向量维度），
        # hidden_size 是隐藏状态维度；batch_first=True 表示输入/输出张量把 batch 放在第 0 维。
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)

    def forward(self, x):
        """前向传播：中文索引序列 -> 句子最后一个真实 token 处的隐藏状态。

        :param x: 中文索引序列，形状 [batch_size, seq_len]
        :return: 语义向量，形状 [batch_size, hidden_size]
        """
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]

        # GRU 前向：output 是每个时间步的输出（全序列），_（下划线）是最后一个时间步的隐状态，这里不需要。
        # 因为 batch 内有 padding，直接取"最后一步"会拿到 padding 位的输出，所以下面自己算真实长度。
        output, _ = self.gru(embed)
        # output.shape: [batch_size, seq_len, hidden_size]

        # 计算每个句子中"非 padding"的真实 token 个数：
        # x != padding_idx 得到布尔张量，True 记 1；sum(dim=1) 沿序列维求和，得到每句有效长度。
        lengths = (x != self.embedding.padding_idx).sum(dim=1)

        # 高级索引取值：
        # torch.arange(output.shape[0]) 生成 [0,1,...,batch_size-1] 的行下标；
        # lengths - 1 是每句最后一个真实 token 的列下标；
        # 两者配对，把每个句子的最后真实隐状态取出来，凑成 [batch_size, hidden_size]。
        last_hidden_state = output[torch.arange(output.shape[0]), lengths - 1]
        # last_hidden_state.shape: [batch_size, hidden_size]
        return last_hidden_state


class TranslationDecoder(nn.Module):
    """解码器：一个词一个词地生成英文译文。

    每个时间步输入"上一个已生成的词"，结合当前隐藏状态，输出"下一个词的词表概率分布"。
    """

    def __init__(self, vocab_size, padding_index):
        """初始化解码器：Embedding + GRU + 全连接输出层。

        :param vocab_size: 目标语言（英文）词表大小
        :param padding_index: <pad> 的索引
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)

        # nn.Linear(in_features, out_features)：全连接层，把隐状态维度映射到词表大小，
        # 输出的是"每个候选词的打分（logits）"，后续交给交叉熵/softmax 转成概率。
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,
                                out_features=vocab_size)

    def forward(self, x, hidden_0):
        """单步前向：一个词 + 当前隐藏状态 -> 下一个词分布 + 更新后的隐藏状态。

        :param x: 当前输入的词索引，形状 [batch_size, 1]
        :param hidden_0: 上一时间步的隐藏状态，形状 [1, batch_size, hidden_size]
                         （第 0 维是 GRU 的层数，这里只有 1 层所以是 1）
        :return: (output, hidden_n)
                 output   下一个词的打分分布，形状 [batch_size, 1, vocab_size]
                 hidden_n 更新后的隐藏状态，形状 [1, batch_size, hidden_size]
        """
        # x.shape: [batch_size, 1]
        # hidden.shape: [1, batch_size, hidden_size]
        embed = self.embedding(x)
        # embed.shape: [batch_size, 1, embedding_dim]

        # 把当前词喂进 GRU，并传入上一时间步的隐状态 hidden_0 作为初始状态；
        # 返回 output（本步输出）和 hidden_n（更新后的隐状态，作为下一步的输入）。
        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape: [batch_size, 1, hidden_size]

        # 全连接层：把隐状态向量映射成词表维度的打分，打分最大的索引即最可能的词。
        output = self.linear(output)
        # output.shape: [batch_size, 1, vocab_size]
        return output, hidden_n


class TranslationModel(nn.Module):
    """完整翻译模型：把编码器和解码器组装在一起。

    内部不定义任何网络层，只作为"容器"持有 encoder 与 decoder，
    好处：训练/保存/加载时只需操作一个模型对象。
    """

    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        """组装编码器与解码器。

        :param zh_vocab_size: 中文词表大小
        :param en_vocab_size: 英文词表大小
        :param zh_padding_index: 中文 <pad> 索引
        :param en_padding_index: 英文 <pad> 索引
        """
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size=zh_vocab_size, padding_index=zh_padding_index)
        self.decoder = TranslationDecoder(vocab_size=en_vocab_size, padding_index=en_padding_index)
