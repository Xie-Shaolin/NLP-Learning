# 1.定义Dataset
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import config


class TranslationDataset(Dataset):
    # 创建对象时触发
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')
    # 调用内置 len() 时触发 ： len(dataset) 或 len(dataloader)
    def __len__(self):
        return len(self.data)
    # 按下标取样本时触发：每当你用索引访问 dataset[i] ，
    # 或者 DataLoader 在 迭代取 batch 时 （每次 for ... in dataloader ），
    # 它都会反复调用 dataset[i] 来拿单条样本。
    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input_tensor, target_tensor


# ========== 数据整理函数 ==========
def collate_fn(batch):
    """
        把一个 batch 的样本对齐到相同长度（padding），供 DataLoader 在取批次时调用。
            :param batch: DataLoader 打包好的批次，是二元组列表 [(input_tensor, target_tensor), ...]
            :return: (input_tensor, target_tensor)，形状均为 [batch_size, seq_len]，
                    不同句子用 padding_value=0（即 <pad> 的索引）补齐到 batch 内最大长度
    
        translation-transformer/data/processed/train.jsonl 里面的数据是这样的：
            {"en":[2,7226,7034,6660,7018,2505,3],"zh":[2024,1373,1388,417,2503,1242]}
        但是为什么 item[0] 的中文呢？ 因为这里面的batch 是来自 __getitem__的方法
            for ... in dataloader 开始取一个 batch
            ├─ 1. DataLoader 里的 Sampler（随机采样器）生成一批下标，如 [17, 3, 88, ...]
            ├─ 2. 对每个下标调用 dataset.__getitem__(i)   ← 这里触发了！
            │        数据集 1000 条、batch_size 64 → 每次取 batch 触发 64 次
            ├─ 3. 把 64 条 (zh, en) 拼成一个 batch 列表
            └─ 4. 交给 collate_fn，pad 成 [64, seq_len] 的定长张量，返回给循环体
        在 __getitem__ 里面 返回的是 (input_tensor, target_tensor)，形状均为 [1, seq_len]
    """
    input_tensors = [item[0] for item in batch] # 中文序列
    target_tensors = [item[1] for item in batch] # 英文序列


    '''
    pad_sequence 是 PyTorch 中 torch.nn.utils.rnn 模块提供的一个函数，用于把 长度不等的张量序列列表 对齐成 一个等长的批张量 。
    torch.nn.utils.rnn.pad_sequence(
        sequences,          # 长度不等的张量列表，如 [shape (5,), (3,), (7,)]
        batch_first=False,  # 输出第 0 维是否为 batch
        padding_value=0.0   # 填充用的数值
    )
    它做的就一件事： 找出所有序列中的最大长度，把短的序列尾部补上 padding_value ，直到和最长的一样长 ，然后把它们堆叠成一个 2D（或更高维）张量。
    import torch
    from torch.nn.utils.rnn import pad_sequence

    a = torch.tensor([1, 2, 3])        # 长度 3
    b = torch.tensor([4, 5])           # 长度 2
    c = torch.tensor([6])              # 长度 1

    out = pad_sequence([a, b, c], batch_first=True, padding_value=0)
    print(out)
    # tensor([[1, 2, 3],
    #         [4, 5, 0],   ← 第 2 条用 0 补到长度 3
    #         [6, 0, 0]])  ← 第 3 条用 0 补到长度 3
    # out.shape = (3, 3)
    '''
    # pad_sequence：把长度不一的张量列表按最长的对齐。
    # batch_first=True 表示返回张量的第 0 维是 batch；
    # padding_value=0 指定用 0（<pad>）填充空缺位置。
    input_tensor = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensor = pad_sequence(target_tensors, batch_first=True, padding_value=0)

    return input_tensor, target_tensor


def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = TranslationDataset(path)
    # DataLoader 参数说明：
    #   batch_size   每次取多少条样本
    #   shuffle=True 每个 epoch 都打乱顺序（训练需要；测试一般不打乱）
    #   collate_fn   自定义"如何把多条样本打包成一个 batch"，这里用来 padding
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor)  # input_tensor.shape: [batch_size, seq_len]
        print(target_tensor)  # target_tensor.shape : [batch_size]
        break
