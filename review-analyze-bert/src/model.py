import torch
from torch import nn
from transformers import AutoModel

import config


class ReviewAnalyzeModel(nn.Module):
    """
    基于 BERT 的评论情感分析模型。

    结构：BERT 编码器 + 一个线性分类层。
    取 BERT 输出的 [CLS] 向量（句子整体语义表示），
    经过线性层映射为单个标量 logits，再交由损失函数做二分类。
    """

    def __init__(self):
        """初始化模型：加载预训练 BERT，并构建输出维度为 1 的分类头。"""
        super().__init__()
        # 加载本地预训练的 bert-base-chinese 模型（不含分类头）
        # self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
        self.bert = AutoModel.from_pretrained("google-bert/bert-base-chinese")
        # 分类头：把 BERT 的 hidden_size 维向量压缩为 1 维 logits
        # nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        # in_features=self.bert.config.hidden_size：输入特征维度，即每个样本输入这个层的向量长度
        # out_features=1： 输出特征维度，即每个样本经过该层后的向量长度
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        前向传播。
        :param input_ids: 输入 token 的 ID 序列，shape: [batch_size, seq_len]
        :param attention_mask: 注意力掩码，标记哪些位置是有效 token，shape: [batch_size, seq_len]
        :param token_type_ids: 分段 ID（用于区分句子对），shape: [batch_size, seq_len]
        :return: 每个样本的 logits，shape: [batch_size]
        """
        # shape: [batch_size, seq_len]
        # BERT 编码，output 包含 last_hidden_state 与 pooler_output 等
        output = self.bert(input_ids, attention_mask, token_type_ids)

        # 取最后一层隐藏状态：shape: [batch_size, seq_len, hidden_size]
        last_hidden_state = output.last_hidden_state

        # 取每个序列第一个 token（[CLS]）对应的向量作为整句表示：
        # cls_hidden_state.shape: [batch_size, hidden_size]
        '''
            output 里面 有 自己的 pooler_output，
            这是一个[CLS] token向量经过 Linear(hidden, hidden) + tanh 变换后的结果
            也代表了整句，但是作者没有用，主要是后面作者自己用了 linear + sigmoid
        '''
        cls_hidden_state = last_hidden_state[:, 0, :]

        # 线性分类并去掉最后一维：output.shape: [batch_size]
        output = self.linear(cls_hidden_state).squeeze(-1)
        return output
