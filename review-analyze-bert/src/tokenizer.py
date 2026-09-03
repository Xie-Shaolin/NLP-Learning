import jieba
from tqdm import tqdm
import config


class JiebaTokenizer:
    """
    基于 jieba 分词的自定义分词器。

    提供词表构建、文本分词、文本到 ID 的编码，以及词表读写能力。
    注意：本项目实际训练/预测使用的是 HuggingFace 的 bert-base-chinese 分词器，
    本类保留用于早期自建词表的方案，或作为自定义词表工具使用。
    """

    # 两个特殊 token 的字符串表示
    unk_token = '<unk>'   # 未知词占位符（词表中不存在的词映射到它）
    pad_token = '<pad>'   # 填充占位符（用于把序列补齐到固定长度）

    def __init__(self, vocab_list):
        """
        根据词表初始化分词器，构建词与索引的双向映射。
        :param vocab_list: 词表列表，每个元素是一个词（token）。
        """
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        # 词 -> 索引 的映射，用于编码时把词转为 ID
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        # 索引 -> 词 的映射，用于解码时把 ID 转回词
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        # 提前取出特殊 token 的索引，编码时直接引用
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]

    @staticmethod
    def tokenize(text):
        """使用 jieba 对文本做分词，返回词列表（jieba.lcut 默认精确模式）。"""
        return jieba.lcut(text)

    def encode(self, text, seq_len):
        """
        将文本分词并编码为固定长度的 ID 列表。
        :param text: 输入文本。
        :param seq_len: 目标序列长度。
        :return: 长度为 seq_len 的 token ID 列表，超出截断、不足则用 pad 填充。
        """
        # 先分词得到词列表
        tokens = self.tokenize(text)

        # 截取或填充到指定的长度
        if len(tokens) > seq_len:
            # 超出长度：直接截断，只保留前 seq_len 个词
            tokens = tokens[:seq_len]
        elif len(tokens) < seq_len:
            # 不足长度：在末尾用 pad_token 补齐
            tokens = tokens + [self.pad_token] * (seq_len - len(tokens))

        # 将每个词映射为 ID；词表不存在的词统一映射为 unk_token 的索引
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """
        从语料中构建词表并保存到文件。
        :param sentences: 语料句子集合（可迭代）。
        :param vocab_path: 词表保存路径。
        """
        # 用集合收集所有分词结果，自动去重
        vocab_set = set()
        for sentence in tqdm(sentences, desc="构建词表"):
            vocab_set.update(jieba.lcut(sentence))

        # 词表顺序：先放 pad/unk 两个特殊 token，再放去重后的普通词（过滤空白 token）
        vocab_list = [cls.pad_token, cls.unk_token] + [token for token in vocab_set if token.strip() != '']
        print(f'词表大小:{len(vocab_list)}')

        # 保存词表：每行一个词
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        """从词表文件加载并实例化一个分词器。"""
        # 读取词表文件，按行拆分并去除首尾空白，得到词列表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)


if __name__ == '__main__':
    # 自测：从词表文件加载分词器，打印词表大小和特殊 token
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print(f'词表大小：{tokenizer.vocab_size}')
    print(f'特殊符号：{tokenizer.unk_token}')
    # 编码一段示例文本，验证分词+编码结果
    print(tokenizer.encode("今天天气不错"))
