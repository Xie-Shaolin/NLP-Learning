"""
模型模块
定义基于 LSTM 的情感分析模型。
"""

import torch
from torch import nn
import config


class ReviewAnalyzeModel(nn.Module):
    """
    评论情感分析模型

    网络结构（三层）：
    1. Embedding：把每个词索引映射成稠密向量（词嵌入）
    2. LSTM：处理序列，捕捉评论中词语之间的先后顺序关系
    3. Linear：把 LSTM 输出的隐藏状态映射成一个分数（logits）

    输出是单个分数，经过 sigmoid 后表示"正向评论"的概率。
    """

    def __init__(self, vocab_size, padding_index):
        """
        初始化模型各层

        :param vocab_size: 词表大小（决定 Embedding 层有多少个词向量）
        :param padding_index: 填充符号 <pad> 的索引，让 Embedding 层忽略填充位置
        """
        super().__init__()
        # 词嵌入层：把 vocab_size 个词映射成 EMBEDDING_DIM 维向量
        # padding_idx 指定填充符号索引，该位置不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=padding_index)
        # LSTM 层：输入维度=词嵌入维度，隐藏层维度=HIDDEN_SIZE
        # batch_first=True 表示输入张量形状为 [batch_size, seq_len, feature]
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM,
                            hidden_size=config.HIDDEN_SIZE,
                            batch_first=True)
        # 线性层：把 LSTM 的隐藏状态(HIDDEN_SIZE 维)映射成 1 个分数
        self.linear = nn.Linear(config.HIDDEN_SIZE, 1)

    def forward(self, x):
        """
        前向传播：定义数据从输入到输出的计算流程

        :param x: 输入的词索引序列，shape: [batch_size, seq_len]
        :return: 每个样本的预测分数（logits），shape: [batch_size]
        """
        # 1. 词嵌入：把词索引转成稠密向量
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]

        # 2. 送入 LSTM，得到每个时间步的输出
        # output 是每个时间步的隐藏状态；最后的 (h_n, c_n) 这里用不到，用 _ 忽略
        output, (_, _) = self.lstm(embed)
        # output.shape: [batch_size, seq_len, hidden_size]

        # 3. 提取每个样本"最后一个真实 token"对应的隐藏状态
        #    因为句子可能被 <pad> 填充，不能直接取最后一列，要根据真实长度定位
        # batch_indexes：每个样本对应的行索引 [0, 1, 2, ..., batch_size-1]
        batch_indexes = torch.arange(0, output.shape[0])
        # lengths：每个样本真实（非 pad）的 token 数量
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        # 用行索引 + 真实长度-1 定位最后一个真实 token 的隐藏状态
        last_hidden = output[batch_indexes, lengths - 1]
        # last_hidden.shape: [batch_size, hidden_size]

        # 4. 线性层把隐藏状态映射成 1 个分数，并去掉最后一个维度
        output = self.linear(last_hidden).squeeze(-1)
        # output.shape: [batch_size]
        return output
