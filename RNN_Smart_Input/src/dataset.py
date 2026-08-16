"""
数据集与数据加载器模块
实现 PyTorch Dataset 子类封装输入法训练数据，并提供 DataLoader 获取方法
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class InputMethodDataset(Dataset):
    """
    输入法预测数据集类，继承自 PyTorch Dataset
    
    数据格式：
    每条样本包含：
    - input: 长度为 SEQ_LEN 的词索引序列（历史上下文）
    - target: 单个词索引（需要预测的下一个词）
    
    数据文件格式为 JSONL，每行是一个 JSON 对象，例如：
    {"input": [1, 2, 3, 4, 5], "target": 6}
    
    Attributes:
        data (list): 数据集列表，每个元素是一个包含 'input' 和 'target' 的字典
    """

    def __init__(self, path):
        """
        初始化数据集，从 JSONL 文件中加载数据
        
        Args:
            path (Path): JSONL 格式的数据文件路径
        """
        # 使用 pandas 读取 JSONL 文件（每行一个 JSON 记录）
        # lines=True: 表示每行是一个独立的 JSON 对象
        # orient='records': JSON 记录的格式
        # .to_dict(orient='records'): 将 DataFrame 转换为字典列表，每个字典是一行数据
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        """
        返回数据集的样本总数
        
        Returns:
            int: 数据集中的样本数量
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根据索引获取单个样本，返回输入张量和目标张量
        
        Args:
            index (int): 样本索引，范围为 [0, len(self.data)-1]
        
        Returns:
            tuple: (input_tensor, target_tensor)
                - input_tensor: torch.LongTensor，形状为 [SEQ_LEN]，历史输入序列
                - target_tensor: torch.LongTensor，标量，需要预测的目标词索引
        """
        # 从数据字典中取出输入序列，并转换为 PyTorch long 类型张量
        # long 类型对应整数，适合表示离散的索引
        input_tensor = torch.tensor(self.data[index]['input'], dtype=torch.long)
        
        # 取出目标词索引，转换为 long 类型张量
        target_tensor = torch.tensor(self.data[index]['target'], dtype=torch.long)
        
        return input_tensor, target_tensor


def get_dataloader(train=True):
    """
    获取训练集或测试集的数据加载器
    
    DataLoader 负责将 Dataset 中的样本按批次组织，支持：
    - 自动批处理（batching）
    - 数据打乱（shuffling）
    - 并行加载（多进程）
    
    Args:
        train (bool): 是否加载训练集
                    True -> 加载 train.jsonl（训练集）
                    False -> 加载 test.jsonl（测试集）
    
    Returns:
        DataLoader: PyTorch 数据加载器实例
                    每次迭代返回 (batch_inputs, batch_targets)，形状分别为：
                    - batch_inputs: [batch_size, seq_len]
                    - batch_targets: [batch_size]
    """
    # 根据 train 参数选择训练集或测试集文件路径
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    
    # 初始化数据集实例
    dataset = InputMethodDataset(path)
    
    # 创建数据加载器
    # batch_size: 每个批次包含的样本数
    # shuffle=True: 每个 epoch 开始前打乱数据顺序，提高训练泛化能力
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    """
    主函数：用于单独测试数据集和数据加载器
    运行此文件可快速验证数据加载是否正常工作
    """
    # 获取训练集和测试集的 DataLoader
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    
    # 打印批次数量（len(dataloader) = 总样本数 // batch_size）
    print(len(train_dataloader))
    print(len(test_dataloader))

    # 取出训练集的第一个批次，检查数据形状
    for input_tensor, target_tensor in train_dataloader:
        # input_tensor: 一个批次的输入序列
        # shape: [batch_size, seq_len] -> [64, 5]
        print(input_tensor.shape)
        
        # target_tensor: 对应批次的目标词
        # shape: [batch_size] -> [64]
        print(target_tensor.shape)
        
        # 只查看第一个批次，跳出循环
        break
