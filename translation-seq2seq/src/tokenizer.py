# ========== 分词器模块 ==========
# 负责把句子切成 token（单词/字），并在 token 与数字索引之间互相转换。
# 中文按"字"切分，英文按 NLTK 提供的 Treebank 分词规则切分。

# TreebankWordTokenizer：NLTK（自然语言工具包）中的经典英文分词器，
# 能正确处理标点（如 doesn't -> does n't）、数字格式（$50,000 -> $ 50,000）等。
# TreebankWordDetokenizer：与之配套的"去分词器"，能把 token 列表还原成正常的英文句子（含空格）。
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer

# tqdm：Python 进度条库，把循环包装后会在终端显示实时进度条
from tqdm import tqdm


class BaseTokenizer:
    """分词器基类：定义 4 个特殊 token，并提供编码/建词表/加载词表等通用能力。

    特殊 token 约定（放在词表最前面）：
    - <pad>  填充符：让 batch 中长度不一的句子补齐到相同长度
    - <unk>  未知符：词表外的 token 统一映射成它
    - <sos>  起始符：句子开头标记，告诉解码器"开始生成"
    - <eos>  结束符：句子结尾标记，告诉解码器"到此为止"
    """
    # 类属性：所有实例共享，不需要实例化就能访问（cls.unk_token 或 BaseTokenizer.unk_token 均可）
    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        """初始化：根据传入的词表列表建立"词->索引"和"索引->词"的映射。

        :param vocab_list: 词表列表，如 ['<pad>', '<unk>', ...]
        """
        self.vocab_list = vocab_list            # 原始词表列表
        self.vocab_size = len(vocab_list)       # 词表大小（神经网络输出层维度需要用到）

        # 列表推导式：for index, word in enumerate(vocab_list) 同时拿到下标和元素，
        # 生成 {词: 下标} 字典。enumerate() 是内置函数，返回 (下标, 元素) 的迭代对。
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        # 反查字典：{下标: 词}，用于把模型输出的数字索引还原成文本
        self.index2word = {index: word for index, word in enumerate(vocab_list)}

        # 预先把 4 个特殊 token 的索引存成属性，方便直接取用（如 loss 里忽略 pad）
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    '''
    这里定义了一个"模板方法"模式： 父类定流程，子类定细节 。
    -> list[str]: 声明这个函数"应该返回一个元素为 str 的列表"，如 ['我', '爱', '你'] 。它只是给人和 IDE 看的提示， Python 不会强制校验
    pass 是一条 Python 空语句 ，字面意思就是"什么也不做"。
        它的作用是让函数体 合法地存在但暂时没有内容 。
        因为 Python 语法要求代码块内至少有一条语句，
        如果不写会直接报错：IndentationError: expected an indented block
    类似于java的抽象类abstract，子类必须覆写该方法。
    '''    
    @classmethod
    # @classmethod：类方法装饰器。第一个参数自动绑定为"类"（约定命名为 cls，不是 self）。
    # 它的作用：无需创建实例就能调用（如 ChineseTokenizer.tokenize(text)），
    # 并且能被子类继承/覆写，从而实现"父类定义流程、子类定义细节"的模板方法模式。
    def tokenize(cls, text) -> list[str]:
        """将文本切分成 token 列表。基类只声明不实现（pass 占位），由子类覆写。

        :param text: 原始文本字符串
        :return: token 列表；-> list[str] 是类型注解，表示"返回一个元素为 str 的列表"（Python 3.9+ 语法）
        """
        pass

    def encode(self, text, add_sos_eos=False):
        """把一段文本编码成数字索引列表（文本 -> [1, 5, 8, 2]）。

        :param text: 原始文本
        :param add_sos_eos: 是否在首尾拼接 <sos> 和 <eos>（解码器的输入/目标需要，编码器输入不需要）
        :return: 整数索引列表
        """
        tokens = self.tokenize(text)            # 先切分成 token 列表（调用的是子类覆写后的 tokenize）

        if add_sos_eos:                         # 若要求加起止符
            # 列表拼接：在首尾插入特殊 token
            tokens = [self.sos_token] + tokens + [self.eos_token]

        # 列表推导式：逐个 token 查表转索引。
        # dict.get(key, default)：查不到 key 时返回默认值（这里返回 unk 的索引），不会抛 KeyError。
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """从一批句子构建词表并保存到文件（只统计、不生成实例）。

        :param sentences: 句子列表，如 ['你好', '世界', ...]
        :param vocab_path: 词表保存路径
        """
        vocab_set = set()                       # set 是"无序、自动去重"的集合，用来收集所有不重复的 token
        for sentence in tqdm(sentences, desc="构建词表"):
            # set.update(可迭代对象)：把 tokenize 出来的每个 token 都塞进集合，重复的自动忽略
            vocab_set.update(cls.tokenize(sentence))

        # 词表组成：4 个特殊 token 固定在最前面（保证索引稳定）+ 语料中出现过的 token。
        # 列表推导式加了条件 if token.strip() != ''，即剔除空白字符（strip 去除首尾空格后为空则不要）。
        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + [token for token in vocab_set if
                                                                                    token.strip() != '']
        print(f'词表大小:{len(vocab_list)}')

        # 打开文件写入：with 语句自动管理文件资源（无论是否报错都会自动 close）。
        # 'w' 表示写模式（会清空原文件内容）；encoding='utf-8' 指定编码，避免中文乱码。
        # '\n'.join(vocab_list)：把列表元素用换行符拼成一个大字符串（每个词占一行）。
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        """从词表文件加载并返回一个分词器实例（文件 -> 对象）。

        :param vocab_path: 词表文件路径
        :return: BaseTokenizer（或其子类）实例
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            # 列表推导式：readlines() 按行读取，line.strip() 去掉每行首尾的换行符和空白
            vocab_list = [line.strip() for line in f.readlines()]
        # cls(...) 等价于调用当前类（子类调用时是子类）的构造函数，返回新实例
        return cls(vocab_list)


class ChineseTokenizer(BaseTokenizer):
    """中文分词器：按单个汉字切分（对中文而言"字"即最小语义单元，无需复杂分词算法）。"""

    @classmethod
    def tokenize(cls, text) -> list[str]:
        """覆写基类的 tokenize：中文不做空格切分，直接用 list() 把字符串拆成单字列表。

        list('你好世界') 的结果是 ['你', '好', '世', '界']，
        这是利用了"字符串是可迭代对象"的特性——迭代它会逐个产出字符。
        """
        return list(text)


class EnglishTokenizer(BaseTokenizer):
    """英文分词器：使用 NLTK 的 Treebank 规则分词，并额外提供 decode 方法把索引还原成句子。"""

    # 类属性：TreebankWordTokenizer 的实例，类加载时创建一次，所有实例共用
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text) -> list[str]:
        """覆写基类的 tokenize：调用 NLTK 分词器的 tokenize 方法。

        cls.tokenizer 取的是类属性（该实例挂在本类上），所以即使没有 self 也能访问。
        """
        return cls.tokenizer.tokenize(text)

    def decode(self, indexes):
        """把索引列表还原成英文句子（索引 -> 文本），供预测/评估时展示结果。

        :param indexes: 整数索引列表，如 [3, 12, 7]
        :return: 还原后的英文句子字符串
        """
        # 列表推导式：通过 index2word 反查字典，把每个索引转回原始单词
        tokens = [self.index2word[index] for index in indexes]
        # detokenize 负责加空格/处理标点粘连（如 "don't" 不会被拆成 "don 't"）
        return self.detokenizer.detokenize(tokens)


if __name__ == '__main__':
    # 仅当直接运行本文件时才执行（__name__ 在"作为主程序"时为 '__main__'；被 import 时是模块名）。
    # 这是一段测试代码，演示 Treebank 分词/去分词效果。
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    # tokenize 对数字、百分号、美元符号的处理：'On a $50,000 mortgage of 30 years at 8 percent...'
    word_list = tokenizer.tokenize(
        'On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88.')
    print(word_list)
    # 去分词：把 token 列表重新拼成带空格的句子，验证是否无损还原
    print(detokenizer.detokenize(word_list))
