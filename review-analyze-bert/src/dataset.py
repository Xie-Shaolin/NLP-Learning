# 1.定义Dataset：本项目使用 HuggingFace datasets 提供的 Arrow 数据集，
#    因此无需再自定义 torch.utils.data.Dataset，直接通过 load_from_disk 加载即可。
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import config


# 2. 提供一个获取 dataloader 的方法
def get_dataloader(train=True):
    """
    加载处理好的数据集并包装成 DataLoader，供训练/评估/预测使用。
    :param train: 为 True 时加载训练集，否则加载测试集。
    :return: 返回一个按批次迭代的 DataLoader，batch 中每个字段都是 torch 张量。
    """
    # 根据 train 标志选择训练集或测试集子目录
    path = str(config.PROCESSED_DATA_DIR / ('train' if train else 'test'))
    # 从磁盘加载 HuggingFace 数据集
    dataset = load_from_disk(path)
    # 将数据集的每个字段统一转换为 torch 张量，便于 DataLoader 直接使用
    dataset.set_format(type='torch')
    # shuffle=True：训练/评估时打乱样本顺序，避免模型学到固定顺序的偏差
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    # 本模块作为脚本运行时，用于自测：分别加载训练集和测试集
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    # 取第一个 batch，打印每个字段的 key 和对应张量形状，验证数据格式是否正确
    for batch in train_dataloader:
        for k, v in batch.items():
            print(k, '->', v.shape)
        break
