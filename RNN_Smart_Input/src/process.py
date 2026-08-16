"""
数据预处理模块
负责将原始对话语料处理为模型训练所需的格式，包括：
- 数据读取与抽样
- 句子提取
- 数据集划分
- 词表构建
- 训练/测试样本生成与保存
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tokenizer import JiebaTokenizer
import config


def build_dataset(sentences, tokenizer):
    """
    sentences = [
        "今天天气不错",       # 4个词
        "你好",               # 2个词
        "深度学习非常有趣"     # 6个词
    ]

    根据句子列表构建模型训练/测试数据集
    
    核心思想：滑动窗口法
    对每个句子，从左到右滑动一个长度为 SEQ_LEN 的窗口：
    - 窗口内的 SEQ_LEN 个词作为输入（input）
    - 窗口后紧邻的下一个词作为预测目标（target）
    
    例如：句子分词后为 [w1, w2, w3, w4, w5, w6, w7]，SEQ_LEN=5
    则生成样本：
    1. input=[w1,w2,w3,w4,w5], target=w6
    2. input=[w2,w3,w4,w5,w6], target=w7
    
    Args:
        sentences (list): 句子列表，每个元素是一个字符串句子
        tokenizer (JiebaTokenizer): 已加载好词表的分词器实例，用于编码
    
    Returns:
        list: 数据集列表，每个元素是字典格式：
            {'input': [idx1, idx2, ..., idx_SEQ_LEN], 'target': idx_next}
    """
    # Step 1: 将所有句子编码为词索引序列
    # tokenizer.encode(sentence) 返回词索引列表
    indexed_sentences = [tokenizer.encode(sentence) for sentence in sentences]
    '''
    indexed_sentences = [
        [12, 34, 56, 78],          # 长度 4（"今天/天气/不错"？不对，实际分词结果对应索引）
        [90, 21],                   # 长度 2
        [43, 65, 87, 98, 11, 22]    # 长度 6
    ]
    '''
    # 初始化数据集列表，用于存储所有训练/测试样本
    dataset = []
    
    # 数据格式示例：[{'input':[1,2,3,4,5],'target':5},{'input':[2,3,4,5,6],'target':7}]
    
    # Step 2: 遍历每个编码后的句子，用滑动窗口生成样本
    # tqdm 显示处理进度
    for sentence in tqdm(indexed_sentences, desc="构建数据集"):
        # 句子分词索引： [w1, w2, w3] range(3-5) = range(-2) → 空 -> 太短，直接跳过
        # 滑动窗口范围：从位置 0 到位置 len(sentence) - SEQ_LEN - 1
        # 确保每个窗口后面还有至少一个词作为 target
        # range(负数) 是空循环，正好让代码自动跳过太短的句子，不需要额外的边界判断。
        for i in range(len(sentence) - config.SEQ_LEN):
            # 截取窗口内的 SEQ_LEN 个词索引作为输入
            input = sentence[i:i + config.SEQ_LEN]
            # 窗口后面紧邻的词索引作为预测目标
            target = sentence[i + config.SEQ_LEN]
            # 将样本添加到数据集
            dataset.append({'input': input, 'target': target})
    
    return dataset


def process():
    """
    数据预处理主流程
    
    执行以下步骤：
    1. 读取原始对话数据文件并抽样
    2. 从对话中提取纯文本句子
    3. 按 8:2 比例划分训练集和测试集
    4. 基于训练集构建词表并保存
    5. 构建训练集样本并保存
    6. 构建测试集样本并保存
    整个数据预处理阶段（ process.py ）只负责把文本转成 整数索引 ，
    真正的 词向量（Embedding） 是在 模型训练时 （ train.py + model.py ）才动态生成的。
    """
    print("开始处理数据")
    
    # Step 1: 读取原始数据文件
    # 读取 JSONL 格式的对话数据
    # lines=True: 每行一个 JSON 记录
    # orient="records": 按记录格式解析
    # sample(frac=0.01): 随机抽取 1% 的数据用于演示/训练（可根据需要调整比例）
    df = pd.read_json(config.RAW_DATA_DIR / "synthesized_.jsonl", lines=True,
                    orient="records").sample(frac=0.01)
    '''
        {"topic": "校园生活分享", 
        "user1": "李欣怡", 
        "user2": "杨欢", 
        "dialog": ["user1：杨欢，最近校园里有什么新鲜事吗？", 
                    "user2：嗨，李欣怡！我们学校刚刚举办了一次科技节，很多学生展示了他们的发明。", 
                    "user1：听起来好有趣！我这边的学校正在筹备一场文化节，主要是推广传统文化。", 
                    "user2：文化节听起来也很棒。你们会做哪些活动呢？"]
        }
    '''
    # Step 2: 从对话数据中提取纯文本句子
    sentences = []
    # 遍历每一条对话记录
    for dialog in df['dialog']:
        # 每条对话包含多个句子（如：发话人和回复）
        for sentence in dialog:
            # 句子格式示例："用户：你好"
            # split('：')[1] 用中文冒号分割，取后半部分（纯文本内容）
            # 丢弃说话人信息，只保留文本内容
            sentences.append(sentence.split('：')[1])
    print(f'句子总数:{len(sentences)}')

    # Step 3: 划分训练集和测试集
    # test_size=0.2: 20% 作为测试集，80% 作为训练集
    # train_test_split 会自动随机打乱数据后划分
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)

    # Step 4: 构建词表：分词
    # 仅基于训练集构建词表（避免数据泄露，模拟真实场景）
    # 将词表保存到 models/vocab.txt
    JiebaTokenizer.build_vocab(train_sentences, config.MODELS_DIR / 'vocab.txt')

    # Step 6: 加载分词器，构建训练集
    # 从已保存的词表文件加载分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    # 使用滑动窗口法构建训练集样本
    train_dataset = build_dataset(train_sentences, tokenizer)

    # Step 7: 保存训练集到 JSONL 文件
    # orient='records': 每行一条记录
    # lines=True: 输出为 JSONL 格式（每行一个 JSON 对象）
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)

    # Step 8: 构建测试集（使用同一分词器，与训练集词表保持一致）
    test_dataset = build_dataset(test_sentences, tokenizer)

    # Step 9: 保存测试集到 JSONL 文件
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)

    print("数据处理完成")


if __name__ == '__main__':
    """
    主函数：执行数据预处理流程
    直接运行此文件即可开始处理数据
    """
    process()
