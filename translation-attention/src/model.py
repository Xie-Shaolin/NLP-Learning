import torch
from torch import nn
import config


# ========== 注意力机制（attention 版相对 seq2seq 的核心提升点） ==========
# 在基础 seq2seq 中，编码器把整句中文压缩成一个固定向量（context vector），
# 解码器每生成一个词都只看这个"压缩后的"向量——句子一长，信息会被稀释，翻译质量下降。
# 注意力机制改进了这一点：解码器每生成一个词时，不再只看固定向量，
# 而是对编码器"每个时间步"的输出做一次"加权求和"，让模型自己学会"当前该重点看源句的哪几个词"。
class Attention(nn.Module):
    def forward(self, decoder_hidden, encoder_outputs):
        """计算当前解码步的上下文向量（context vector）。

        :param decoder_hidden: 解码器当前步的输出，形状 [batch_size, 1, hidden_size]
        :param encoder_outputs: 编码器每个时间步的输出，形状 [batch_size, src_seq_len, hidden_size]
        :return: 加权求和后的上下文向量，形状 [batch_size, 1, hidden_size]
        """
        # 第 1 步：算"注意力分数"（打分）。
        # encoder_outputs.transpose(1, 2) 把 [batch, src_len, hidden] 转成 [batch, hidden, src_len]；
        # transpose 的作用是：把 矩阵 的 第1维和第2维对调。
        # src_len 就是「源语言句子的长度」，即编码器输入的 中文句子被切分成多少个 token（字） 。
        # torch.bmm 是"批量矩阵乘法"（batch matrix multiply），对 batch 里每个样本分别做矩阵乘：
        #   [batch, 1, hidden] × [batch, hidden, src_len] = [batch, 1, src_len]
        # 结果 attention_scores[b, 0, j] 就是"当前解码步"与"编码器第 j 个词"的相关程度（点积打分）。
        '''
            decoder_hidden 来自于 ”output, hidden_n = self.gru(embed, hidden_0)“ 的” output
                decoder_hidden：是解码器得到的当前步骤的隐藏状态（解码器是 单步 的，一次只喂「一个词」：）
                encoder_outputs：是编码器所有时间步的隐藏状态
            decoder_hidden 和 encoder_outputs 做点积
        '''
        attention_scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))

        # 第 2 步：softmax 归一化成"权重"。
        # dim=-1 沿最后一个维（src_len 维）做 softmax，让每个源词对应的分数变成 0~1 之间、
        # 且加起来等于 1 的概率分布——这就是"注意力权重"，数值越大代表越该关注那个词。
        '''因为解码器一次只计算一个词，所以这步得到的就是当前时间步的注意力权重。 '''
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 第 3 步：按权重对编码器输出做加权求和。
        #   [batch, 1, src_len] × [batch, src_len, hidden] = [batch, 1, hidden]
        # 相当于把"所有源词的隐状态"按注意力权重加权平均，得到聚焦后的上下文向量。
        return torch.bmm(attention_weights, encoder_outputs)


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=config.EMBEDDING_DIM,
                                    padding_idx=padding_index)

        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                        hidden_size=config.HIDDEN_SIZE,
                        batch_first=True)

    def forward(self, x):
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]
        output, _ = self.gru(embed)
        # output.shape: [batch_size, seq_len, hidden_size]

        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        last_hidden_state = output[torch.arange(output.shape[0]), lengths - 1]
        # last_hidden_state.shape: [batch_size, hidden_size]

        # 【与 seq2seq 的差异】这里多返回了 output（编码器每个时间步的输出）。
        # 基础 seq2seq 的编码器只返回 last_hidden_state（一个压缩后的固定向量）；
        # attention 版因为解码器要做"注意力加权求和"，需要拿到"每个源词"的隐状态，
        # 所以把完整序列 output 也一并返回，供解码器计算 attention 使用。
        return output, last_hidden_state


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=config.EMBEDDING_DIM,
                                    padding_idx=padding_index)

        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                        hidden_size=config.HIDDEN_SIZE,
                        batch_first=True)

        # 【与 seq2seq 的差异】新增注意力模块：解码器每步用它计算"聚焦后"的上下文向量。
        self.attention = Attention()

        # 【与 seq2seq 的差异】线性层输入维度从 hidden_size 变成 2 * hidden_size。
        # 因为最终预测时，会把"解码器本步输出"和"注意力得到的上下文向量"拼接（cat）在一起，
        # 拼接后维度翻倍（hidden + hidden = 2*hidden），所以线性层要按这个更大的维度接收输入。
        self.linear = nn.Linear(in_features=2 * config.HIDDEN_SIZE,
                                out_features=vocab_size)

    def forward(self, x, hidden_0, encoder_outputs):
        # x.shape: [batch_size, 1]
        # hidden.shape: [1, batch_size, hidden_size]
        # 【与 seq2seq 的差异】新增第 3 个参数 encoder_outputs：编码器全序列输出，供注意力计算使用。
        embed = self.embedding(x)
        # embed.shape: [batch_size, 1, embedding_dim]
        ''' 
            注意： 这里输入的是 hidden_0，也就是其实是原来的 seq2seq 的 context_vector，也就是 last_hidden_state。 
            重要理解点：在 train.py 里面的代码是：for i in range(seq_len): ... decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            所以，解码器是 单步 的，一次只喂「一个词」
            所以output.shape: [batch_size, 1, hidden_size]
        '''
        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape: [batch_size, 1, hidden_size]

        # 应用注意力机制（output, encoder_outputs）
        # 【与 seq2seq 的差异】用当前步的 GRU 输出 output 作为"查询"，
        # 在编码器全序列 encoder_outputs 上做加权求和，得到聚焦后的上下文向量。
        # 相比 seq2seq 全程复用同一个固定向量，这里每一步的 context_vector 都是"动态、按需"的。
        context_vector = self.attention(output, encoder_outputs)
        # context_vector.shape: [batch_size, 1, hidden_size]

        # 融合信息
        # 【与 seq2seq 的差异】把"解码器本步输出"和"注意力上下文向量"沿最后一维拼接，
        # 让模型同时看到"该生成什么词（output）"和"该关注源句哪里（context_vector）"。
        combined = torch.cat([output, context_vector], dim=-1)
        # combined.shape: [batch_size, 1, hidden_size * 2]

        output = self.linear(combined)
        # output.shape: [batch_size, 1, vocab_size]
        return output, hidden_n


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size=zh_vocab_size, padding_index=zh_padding_index)
        self.decoder = TranslationDecoder(vocab_size=en_vocab_size, padding_index=en_padding_index)
