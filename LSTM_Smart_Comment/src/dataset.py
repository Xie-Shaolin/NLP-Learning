"""
数据集模块
定义 PyTorch 的 Dataset 和 DataLoader，用于加载预处理后的数据并供模型训练/评估使用。
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class ReviewAnalyzeDataset(Dataset):
    """
    评论情感分析数据集

    从 jsonl 文件中读取数据，每条数据包含：
    - review：已经编码成数字索引序列的评论
    - label：情感标签（0 表示负向，1 表示正向）

    继承 PyTorch 的 Dataset，实现 __len__ 和 __getitem__ 即可被 DataLoader 使用。
    """

    def __init__(self, path):
        """
        初始化数据集并加载数据

        :param path: jsonl 数据文件路径
        """
        # 读取 jsonl 文件（orient='records' 表示每行是一个记录），再转成字典列表，方便按索引取值
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        """
        返回数据集样本总数

        :return: 样本数量
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根据索引返回一条样本（输入张量 + 目标标签）

        :param index: 样本索引
        :return: (input_tensor, target_tensor)
                 input_tensor 是评论的索引序列，shape: [seq_len]
                 target_tensor 是情感标签，shape: [1]（标量）
        """
        # 评论的索引序列 -> 转成 long 类型张量（索引必须是整数）
        input_tensor = torch.tensor(self.data[index]['review'], dtype=torch.long)
        # 情感标签 -> 转成 float 类型张量（用于计算损失）
        target_tensor = torch.tensor(self.data[index]['label'], dtype=torch.float)
        return input_tensor, target_tensor


def get_dataloader(train=True):
    """
    获取数据加载器（DataLoader）

    :param train: True 表示加载训练集，False 表示加载测试集
    :return: 一个可迭代的 DataLoader，每次返回一个 batch 的 (输入, 标签)
    """
    # 根据 train 参数选择对应的数据文件
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    # 用数据文件构造数据集
    dataset = ReviewAnalyzeDataset(path)
    # 构造 DataLoader：指定批大小，并打乱数据顺序（打乱有助于训练收敛）
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    # 单独运行本文件时，做一次数据加载的冒烟测试
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    # 打印训练集和测试集各有多少个 batch
    print(len(train_dataloader))
    print(len(test_dataloader))

    # 取出第一个 batch 查看张量形状
    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape)  # input_tensor.shape: [batch_size, seq_len]
        print(target_tensor.shape)  # target_tensor.shape : [batch_size]
        break
