# 1.定义Dataset
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import config


# 2. 提供一个获取dataloader的方法
def get_dataloader(train=True):
    path = str(config.PROCESSED_DATA_DIR / ('train' if train else 'test'))
    dataset = load_from_disk(path)
    dataset.set_format(type='torch')
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for batch in train_dataloader:
        for k, v in batch.items():
            print(k,'->',v.shape)
        break
