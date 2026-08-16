"""
中文分词与词表管理模块
基于 Jieba 分词器实现中文文本的分词、编码和解码功能
"""

import jieba
from tqdm import tqdm
import config


class JiebaTokenizer:
    """
    基于 Jieba 的中文分词器类
    
    提供以下核心功能：
    1. 中文分词：将文本切分为词语序列
    2. 文本编码：将文本转换为词索引序列
    3. 词表构建：从语料中构建词表并保存
    4. 词表加载：从已保存的词表文件加载分词器
    
    Attributes:
        unk_token (str): 未知词标记，用于表示词表中不存在的词
        vocab_list (list): 词表列表，按索引顺序存储所有词
        vocab_size (int): 词表大小（包含未知词标记）
        word2index (dict): 词到索引的映射字典，用于快速查询词的编号
        index2word (dict): 索引到词的映射字典，用于从编号还原词
        unk_token_index (int): 未知词标记对应的索引编号
    """
    
    # 未知词标记：当输入词不在词表中时，使用此标记代替
    unk_token = '<unk>'

    def __init__(self, vocab_list):
        """
        初始化 Jieba 分词器
        
        Args:
            vocab_list (list): 词表列表，第一个元素应为未知词标记
                            格式：['<unk>', '词1', '词2', ...]
        """
        # 保存原始词表列表
        self.vocab_list = vocab_list
        
        # 计算词表大小（词的总数）
        self.vocab_size = len(vocab_list)
        
        # 构建词 -> 索引 的映射字典，用于编码时快速查找
        # enumerate 返回 (索引, 词) 元组，转换为 {词: 索引} 形式
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        
        # 构建索引 -> 词 的映射字典，用于解码时还原词
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        
        # 记录未知词标记对应的索引，用于处理未登录词
        self.unk_token_index = self.word2index[self.unk_token]

    @staticmethod
    def tokenize(text):
        """
        静态方法：对输入文本进行分词
        
        使用 Jieba 的精确模式进行分词，将连续文本切分为词语列表
        
        Args:
            text (str): 待分词的中文文本
        
        Returns:
            list: 分词后的词语列表，例如 ['今天', '天气', '不错']
        """
        # jieba.lcut 返回列表形式的分词结果
        return jieba.lcut(text)

    def encode(self, text):
        """
        将文本编码为词索引序列
        
        流程：分词 -> 查词表获取索引 -> 未登录词使用未知词索引
        
        Args:
            text (str): 待编码的中文文本
        
        Returns:
            list: 词索引序列，例如 [123, 456, 789]
                长度与分词后的词语数量相同
        """
        # Step 1: 先对文本进行分词
        tokens = self.tokenize(text)
        
        # Step 2: 遍历每个词，查询其在词表中的索引
        # 使用 dict.get() 方法，当词不在词表中时返回默认的未知词索引
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """
        类方法：从句子列表中构建词表并保存到文件
        
        流程：遍历所有句子 -> Jieba分词 -> 收集所有出现过的词 -> 去重 -> 添加未知词标记 -> 保存
        
        Args:
            sentences (list): 训练语料的句子列表，每个元素是一个字符串句子
            vocab_path (Path): 词表保存的文件路径
        """
        # 使用集合存储词，集合自动去重，效率高
        vocab_set = set()
        
        # 遍历每个句子，进行分词并将分词结果加入词表集合
        # tqdm 显示进度条，desc 为进度条描述文字
        for sentence in tqdm(sentences, desc="构建词表"):
            # jieba.lcut(sentence) 返回分词后的词列表
            # update 将词列表中的所有元素添加到集合中
            vocab_set.update(jieba.lcut(sentence))

        # 构建最终词表：在最前面添加未知词标记，保证索引为 0
        vocab_list = [cls.unk_token] + list(vocab_set)
        
        # 打印词表大小信息，便于确认
        print(f'词表大小:{len(vocab_list)}')

        # 将词表保存到文件，每行一个词
        # '\n'.join(vocab_list) 将词表列表用换行符连接成一个长字符串
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        """
        类方法：从已保存的词表文件加载分词器实例
        
        Args:
            vocab_path (Path): 词表文件路径，每行存储一个词
        
        Returns:
            JiebaTokenizer: 加载好的分词器实例，可直接用于编码/分词
        """
        # 打开词表文件，按行读取所有内容
        with open(vocab_path, 'r', encoding='utf-8') as f:
            # f.readlines() 读取所有行（含换行符）-> 列表 '<unk>\n', '今天\n', '天气\n', '不错\n'
            # [line.strip() for line in ...] 去除每行首尾的空白字符（换行符、空格等）
            # line.strip() 去除每行首尾的换行符
            # 最终结果为 ['<unk>', '今天', '天气', '不错']
            vocab_list = [line.strip() for line in f.readlines()]
        
        # 调用构造函数，传入词表列表，创建分词器实例并返回
        # 构造函数会自动初始化词表的大小、词 -> 索引 的映射字典、索引 -> 词 的映射字典、未知词标记的索引
        return cls(vocab_list)


if __name__ == '__main__':
    """
    主函数：用于单独测试 JiebaTokenizer 的功能
    运行此文件可快速验证分词器是否正常工作
    """
    # 从文件加载词表，创建分词器实例
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    
    # 打印词表基本信息
    print(f'词表大小：{tokenizer.vocab_size}')
    print(f'特殊符号：{tokenizer.unk_token}')
    
    # 测试编码功能：将句子转换为索引序列
    print(tokenizer.encode("今天天气不错"))
