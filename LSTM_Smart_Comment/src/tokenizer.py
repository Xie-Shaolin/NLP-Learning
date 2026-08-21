"""
分词器模块
基于 jieba 实现文本分词，并负责：
1. 将文本切分成词(tokenize)
2. 把词转换成数字索引(encode)
3. 构建词表(build_vocab)与加载词表(from_vocab)
"""

import jieba
from tqdm import tqdm
import config


class JiebaTokenizer:
    """
    基于 jieba 的分词器

    核心职责：
    - 用 jieba 对中文文本分词
    - 维护"词 -> 索引"和"索引 -> 词"的映射关系（词表）
    - 把文本编码成固定长度的数字序列，供模型使用

    特殊符号说明：
    - <unk> 表示未知词（不在词表中的词都映射到这里）
    - <pad> 表示填充符号（把长度不足的句子补齐用）
    """

    # 未知词符号：编码时遇到词表里没有的词，统一用这个符号的索引代替
    unk_token = '<unk>'
    # 填充符号：把长度不足的句子补齐到固定长度
    pad_token = '<pad>'

    def __init__(self, vocab_list):
        """
        根据词表列表初始化分词器

        :param vocab_list: 词表列表，每个元素是一个词
        """
        # 保存词表列表
        self.vocab_list = vocab_list
        # 词表大小（词的总数）
        self.vocab_size = len(vocab_list)
        # 词 -> 索引 的映射，用于编码
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        # 索引 -> 词 的映射，用于解码（把数字还原成词）
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        # 记录 <unk> 对应的索引
        self.unk_token_index = self.word2index[self.unk_token]
        # 记录 <pad> 对应的索引
        self.pad_token_index = self.word2index[self.pad_token]

    @staticmethod
    def tokenize(text):
        """
        对文本进行分词

        :param text: 输入文本
        :return: 分词后的词列表
        """
        # jieba.lcut 返回分词结果列表，例如 "今天天气不错" -> ["今天", "天气", "不错"]
        return jieba.lcut(text)

    def encode(self, text, seq_len):
        """
        把文本编码成固定长度的数字索引序列

        :param text: 输入文本
        :param seq_len: 目标序列长度
        :return: 长度为 seq_len 的整数列表，每个整数是词在词表中的索引
        """
        # 1. 先分词
        tokens = self.tokenize(text)

        # 2. 调整长度到 seq_len
        # 如果词数超过 seq_len，截断多余部分
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        # 如果词数不足 seq_len，用 <pad> 符号补齐
        elif len(tokens) < seq_len:
            tokens = tokens + [self.pad_token] * (seq_len - len(tokens))

        # 3. 把每个词转成索引：词表里没有的词（如生僻词）用 <unk> 的索引代替
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """
        从句子列表中统计词并构建词表，保存到文件

        :param sentences: 句子列表（用于统计出现过的所有词）
        :param vocab_path: 词表保存路径
        """
        # 用集合去重，统计所有句子中出现过的词
        vocab_set = set()
        # tqdm 用于显示进度条
        for sentence in tqdm(sentences, desc="构建词表"):
            # 对每个句子分词，把词加入集合
            vocab_set.update(jieba.lcut(sentence))

        # 词表顺序：先放两个特殊符号，再放普通词（过滤掉空字符串）
        vocab_list = [cls.pad_token, cls.unk_token] + [token for token in vocab_set if token.strip() != '']
        print(f'词表大小:{len(vocab_list)}')

        # 把词表写入文件，每个词占一行
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        """
        从词表文件加载分词器

        :param vocab_path: 词表文件路径
        :return: 构建好的 JiebaTokenizer 实例
        """
        # 读取文件，去掉每行末尾的换行符
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        # 用词表列表构造分词器对象
        return cls(vocab_list)


if __name__ == '__main__':
    # 下面代码仅用于单独运行本文件时做简单测试
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print(f'词表大小：{tokenizer.vocab_size}')
    print(f'特殊符号：{tokenizer.unk_token}')
    print(tokenizer.encode("今天天气不错"))
