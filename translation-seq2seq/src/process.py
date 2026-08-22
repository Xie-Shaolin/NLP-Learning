# ========== 数据预处理模块 ==========
# 读取原始中英平行语料 -> 划分训练/测试集 -> 构建词表 -> 编码成索引序列 -> 保存为 jsonl。
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from tokenizer import EnglishTokenizer, ChineseTokenizer


def process():
    """数据预处理主流程（从原始数据到训练可用的 jsonl 文件）。"""
    print("开始处理数据")
    # 读取文件
    # pandas 读取 TSV 文件（tab 分隔）：
    #   sep='\t'            分隔符是制表符
    #   header=None         文件没有表头行
    #   usecols=[0, 1]      只取前两列（英文、中文）
    #   names=['en', 'zh']  给两列起名字
    #   encoding='utf-8'    指定编码
    #   .dropna()           删除含空值的行（Na 表示缺失值）
    df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt', sep='\t', header=None, usecols=[0, 1], names=['en', 'zh'],
                     encoding='utf-8').dropna()

    # 划分数据集
    # train_test_split：按 test_size=0.2 随机切分，即 80% 训练、20% 测试；
    # 返回两个 DataFrame，分别赋给 train_df 和 test_df。
    train_df, test_df = train_test_split(df, test_size=0.2)

    # 构建词表
    # 只统计训练集的 token（避免测试集"偷看"词表，这是标准做法），保存到 models/ 下的词表文件。
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.MODELS_DIR / 'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.MODELS_DIR / 'en_vocab.txt')

    # 构建Tokenizer
    # 从刚才保存的词表文件加载出分词器实例，用来把文本编码成索引序列。
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')

    # 构建训练集
    # DataFrame.apply：对某一列逐行套用 lambda 函数（匿名函数）。
    # 中文作为编码器输入，不需要加 <sos>/<eos>（add_sos_eos=False）；
    # 英文作为解码器目标，需要加起止符（add_sos_eos=True），用于训练时做"前移错位"监督。
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    # 保存训练集
    # to_json 把 DataFrame 写成 jsonl（每行一个 JSON 对象）：
    # orient='records' 按记录输出，lines=True 一行一条记录。
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)

    # 构建测试集
    # 与训练集处理方式完全一致（注意：测试集用的仍是训练集训练出的词表，保证索引空间一致）。
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    # 保存测试集
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)

    print("处理数据完成")


if __name__ == '__main__':
    # 仅直接运行本文件时执行预处理
    process()
