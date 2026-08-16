"""
输入法模型定义模块
基于 RNN（循环神经网络）实现的中文输入法预测模型
"""

from torch import nn
import config


class InputMethodModel(nn.Module):
    """
    输入法预测模型类
    
    该模型使用 Embedding + RNN + 全连接层的结构，实现根据历史输入序列预测下一个词。
    
    模型结构：
    1. Embedding层：将词索引转换为密集向量表示
    2. RNN层：循环神经网络，捕获序列上下文信息
    3. Linear层：全连接层，将RNN输出映射到词表大小的概率分布
    
    Attributes:
        embedding (nn.Embedding): 词嵌入层
        rnn (nn.RNN): 循环神经网络层
        linear (nn.Linear): 输出全连接层
    """

    def __init__(self, vocab_size):
        """
        初始化输入法模型
        
        Args:
            vocab_size (int): 词表大小，决定Embedding层输入维度和全连接层输出维度
        """
        super().__init__()
        
        # 词嵌入层：将离散的词索引转换为连续的向量表示
        # num_embeddings: 词表大小
        # embedding_dim: 每个词向量的维度
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM
        )

        '''
            nn.Embedding 的词向量是 从随机值开始，随训练不断更新 的"可学习参数"，
            不是像 Word2Vec/GloVe 那样的预训练词向量。
        '''
        # RNN循环神经网络层：处理序列数据，捕捉上下文依赖关系
        # input_size: 输入特征维度，与embedding维度一致
        # hidden_size: 隐藏状态维度，控制RNN的记忆容量
        # batch_first=True: 输入输出格式为 [batch_size, seq_len, feature_dim]
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True
        )

        # 全连接输出层：将RNN最后一步的隐藏状态映射到词表空间，输出每个词的预测分数
        # in_features: 输入维度，与RNN隐藏层大小一致
        # out_features: 输出维度，等于词表大小（每个词对应一个分数）
        self.linear = nn.Linear(
            in_features=config.HIDDEN_SIZE,
            out_features=vocab_size
        )

    def forward(self, x):
        """
        前向传播函数，定义模型的计算流程
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len]
                            batch_size: 批次大小，一次性处理的样本数
                            seq_len: 序列长度，即历史输入的词数
        
        Returns:
            torch.Tensor: 预测输出张量，形状为 [batch_size, vocab_size]
                        每个元素是对应词的预测分数（未归一化的logits）
        """
        # x.shape: [batch_size, seq_len] - 输入为词索引序列
        
        # Step 1: 词嵌入 - 将词索引转换为向量表示
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]
        
        # Step 2: RNN编码 - 处理序列，获取每个时间步的隐藏状态
        # output包含所有时间步的隐藏状态，第二个返回值是最后一步的隐藏状态（此处忽略）
        output, _ = self.rnn(embed)
        # output.shape: [batch_size, seq_len, hidden_size]
        
        # Step 3: 提取最后一个时间步的隐藏状态
        # 只需要序列最后一个位置的输出，用于预测下一个词
        # 切片操作 [:, -1, :] 表示：所有批次、最后一个序列位置、所有特征维度
        last_hidden_state = output[:, -1, :]
        # last_hidden_state.shape: [batch_size, hidden_size]
        
        # Step 4: 全连接层映射 - 将隐藏状态转换为词表维度的预测分数
        output = self.linear(last_hidden_state)
        # output.shape: [batch_size, vocab_size] - 每行对应一个样本对所有词的预测分数
        
        return output
