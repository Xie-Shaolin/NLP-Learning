import math

import torch
from torch import nn
import config


# class PositionEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super().__init__()
#         self.d_model = d_model
#         self.max_len = max_len
#
#         pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)  # pos.shape: (max_len, 1)
#         _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float)  # _2i.shape: (d_model/2,)
#         div_term = torch.pow(10000, _2i / self.d_model)
#
#         sins = torch.sin(pos / div_term)  # sins.shape: (max_len, d_model/2)
#         coss = torch.cos(pos / div_term)  # coss.shape: (max_len, d_model/2)
#
#         pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float)  # pe.shape: (max_len, d_model)
#
#         pe[:, 0::2] = sins
#         pe[:, 1::2] = coss
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         seq_len = x.size(1)
#         return x + self.pe[:seq_len]


class PositionEncoding(nn.Module):
    """位置编码：给每个 token 生成一个固定向量，叠加到词向量上。

    【与 attention 版的差异】attention 版用 GRU 逐词递归，天然携带先后顺序；
    Transformer 没有递归结构、并行处理整句，无法感知 token 顺序，因此必须手动注入位置信息，
    否则 "我爱中国" 和 "中国爱我" 会得到完全相同的表示。
    """

    def __init__(self, max_len, dim_model):
        super().__init__()
        # 用 sin/cos 交替编码位置：偶数维用 sin、奇数维用 cos。
        # 同一维度随位置变化、不同维度频率不同，从而让每个位置得到唯一的编码向量。
        pe = torch.zeros([max_len, dim_model], dtype=torch.float)
        for pos in range(max_len):
            for _2i in range(0, dim_model, 2):
                pe[pos, _2i] = math.sin(pos / (10000 ** (_2i / dim_model)))
                pe[pos, _2i + 1] = math.cos(pos / (10000 ** (_2i / dim_model)))

        # register_buffer：把 pe 注册为 buffer——不参与反向传播，但会随模型一起保存/加载、迁移设备。
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape: [batch_size, seq_len, dim_model]
        seq_len = x.shape[1]
        # 这里的 pe 来自于 init 中的 pe
        # 这里的 part_pe 是截取 pe 中的前 seq_len 个元素
        part_pe = self.pe[0:seq_len]
        # part_pe.shape: [seq_len, dim_model]
        # 逐元素加到词向量上（[seq_len, dim_model] 自动广播到整个 batch）。
        return x + part_pe


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        # nn.Embedding 本质是一张 可训练的查找表（Lookup Table） ：输入一个词在词表中的 整数下标 ，它返回对应的 稠密向量 。
        # 中英文各自独立的 embedding 层：源语言和目标语言词表不同，需要分开查表。
        '''
            padding_idx 的作用？
            padding_idx 的作用是告诉 Embedding 层：" 下标为这个值的 token 是填充符号，它不携带任何语义，也不需要参与学习 "
        '''
        self.zh_embedding = nn.Embedding(num_embeddings=zh_vocab_size, # 词表大小（有多少个词）
                                        embedding_dim=config.DIM_MODEL, # 每个词的向量维度 = 128
                                        padding_idx=zh_padding_index) # padding 符号的专属下标

        self.en_embedding = nn.Embedding(num_embeddings=en_vocab_size,
                                        embedding_dim=config.DIM_MODEL,
                                        padding_idx=en_padding_index)

        # 位置编码：源、目标语言共享同一套（位置信息与语言无关）。
        self.position_encoding = PositionEncoding(config.MAX_SEQ_LENGTH, config.DIM_MODEL)

        # 核心：PyTorch 自带的 Transformer（内含编码器 + 解码器）。
        # 【与 attention 版的差异】attention 版是「GRU 编码器 + GRU 解码器 + 自定义 Attention」；
        # transformer 版直接复用 nn.Transformer，其内部已实现多头自注意力、前馈网络等全部组件。
        # batch_first=True 让输入统一为 [batch, seq_len, dim] 的形状。
        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                        nhead=config.NUM_HEADS,
                                        num_encoder_layers=config.NUM_ENCODER_LAYERS,
                                        num_decoder_layers=config.NUM_DECODER_LAYERS,
                                        batch_first=True)

        # 把 Transformer 输出投影到英文词表维度，得到每个位置对每个英文词的打分（logits）。
        self.linear = nn.Linear(in_features=config.DIM_MODEL, out_features=en_vocab_size)

    def forward(self, src, tgt, src_pad_mask, tgt_mask):
        # 一次前向完成「编码 + 解码」。
        # 【与 attention 版的差异】attention 版训练时需要在 train.py 里手动 for 循环逐词解码；
        # transformer 版直接把整句目标序列喂进去并行计算。
        memory = self.encode(src, src_pad_mask)
        return self.decode(tgt, memory, tgt_mask, src_pad_mask)

    def encode(self, src, src_pad_mask):
        # 编码器：中文 -> 上下文表示 memory。
        # src.shape = [batch_size, src_len]
        # src_pad_mask.shape = [batch_size, src_len]
        embed = self.zh_embedding(src)
        # embed.shape = [batch_size, src_len, dim_model]
        embed = self.position_encoding(embed)

        # src_key_padding_mask 标记中文里哪些位置是 <PAD>，让编码器在自注意力时忽略这些填充位置。
        memory = self.transformer.encoder(src=embed, src_key_padding_mask=src_pad_mask)
        # memory.shape: [batch_size, src_len, d_model]

        return memory

    def decode(self, tgt, memory, tgt_mask, memory_pad_mask):
        # 解码器：英文目标序列 + 编码器输出 memory -> 英文 logits。
        # tgt.shape: [batch_size, tgt_len]
        embed = self.en_embedding(tgt)
        embed = self.position_encoding(embed)
        # embed.shape: [batch_size, tgt_len, dim_model]

        # tgt_mask 是「下三角掩码」，保证解码第 i 个词时只能看到第 0~i-1 个词（防止偷看未来答案）。
        # memory_key_padding_mask 同样用于忽略中文侧的 <PAD>。
        output = self.transformer.decoder(tgt=embed, memory=memory,
                                        tgt_mask=tgt_mask, memory_key_padding_mask=memory_pad_mask)
        # output.shape: [batch_size, tgt_len, dim_model]

        outputs = self.linear(output)
        # outputs.shape: [batch_size, tgt_len, en_vocab_size]
        return outputs
