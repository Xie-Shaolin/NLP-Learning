"""
数据预处理模块
负责：读取原始 CSV 数据 -> 划分训练/测试集 -> 构建词表 -> 把文本编码成索引 -> 保存处理后的数据。
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer

import config


def process():
    """
    数据预处理主流程
    """
    print('开始处理数据')

    # 1. 读取原始数据：只取 label 和 review 两列，并丢弃空值行
    df = pd.read_csv(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv', usecols=['label', 'review'],
                     encoding='utf-8').dropna()

    # 2. 划分训练集和测试集（80% 训练，20% 测试）
    #    stratify=df['label'] 保证训练集和测试集中正负样本比例一致
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    # 3. 用训练集的评论构建词表并保存（只统计训练集，避免数据泄露）
    JiebaTokenizer.build_vocab(train_df['review'].tolist(), config.MODELS_DIR / 'vocab.txt')

    # 4. 从词表文件加载分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')

    # 5. 计算合适的序列长度（可选，注释保留用于调试）
    # print(train_df['review'].apply(lambda x: len(tokenizer.tokenize(x))).quantile(0.95))

    # 6. 把训练集的每条评论编码成固定长度的索引序列
    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))

    # 7. 保存处理后的训练集（jsonl 格式，每行一条记录）
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)

    # 8. 把测试集的每条评论编码成固定长度的索引序列
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))

    # 9. 保存处理后的测试集
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)

    print('数据处理完成')


if __name__ == '__main__':
    process()
