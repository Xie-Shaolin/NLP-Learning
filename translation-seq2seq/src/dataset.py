# ========== 数据集模块 ==========
# 把处理好的 jsonl 文件封装成 PyTorch 的 Dataset 和 DataLoader，
# 供训练/评估/预测循环批量取数据。
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import config


class TranslationDataset(Dataset):
    """翻译数据集类。

    继承 torch.utils.data.Dataset，必须实现 __len__ 和 __getitem__ 两个魔法方法，
    DataLoader 才能知道"一共有多少条数据"以及"怎么取某一条数据"。
    """

    def __init__(self, path):
        """读取 jsonl 文件，把每条记录存成字典列表。

        :param path: jsonl 文件路径（train.jsonl 或 test.jsonl）
        """
        # pandas 读取 jsonl：lines=True 表示每行是一个 JSON 对象，orient='records' 表示按记录解析。
        # .to_dict(orient='records') 再把 DataFrame 转成"字典列表"，
        # 形如 [{'zh': [1, 3, 5], 'en': [2, 4, 6]}, ...]——每个元素是一个样本。
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        """返回数据集样本总数（DataLoader 用来计算 epoch 长度）。"""
        return len(self.data)

    def __getitem__(self, index):
        """按索引返回一个样本，即 (中文索引序列, 英文索引序列)。

        :param index: 样本下标
        :return: (input_tensor, target_tensor)，分别是形状 [src_seq_len] 和 [tgt_seq_len] 的 long 型张量
        """
        # torch.tensor(数据, dtype=torch.long)：把 Python 列表转成张量，
        # dtype=torch.long 指定为 64 位整数型（Embedding 层要求输入是整数索引）。
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)   # 中文（源语言）
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)  # 英文（目标语言）
        return input_tensor, target_tensor


# ========== 数据整理函数 ==========
def collate_fn(batch):
    """把一个 batch 的样本对齐到相同长度（padding），供 DataLoader 在取批次时调用。

    :param batch: DataLoader 打包好的批次，是二元组列表 [(input_tensor, target_tensor), ...]
    :return: (input_tensor, target_tensor)，形状均为 [batch_size, seq_len]，
            不同句子用 padding_value=0（即 <pad> 的索引）补齐到 batch 内最大长度
    """
    # 列表推导式：把每个样本的 input 拆出来组成新列表
    input_tensors = [item[0] for item in batch] # 中文（源语言）
    target_tensors = [item[1] for item in batch] # 英文（目标语言）

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
    """按数据集类型（训练/测试）构建并返回 DataLoader。

    :param train: True 返回训练集加载器，False 返回测试集加载器
    :return: torch.utils.data.DataLoader 实例
    """
    # 条件表达式：train 为真时取 train.jsonl，否则取 test.jsonl
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = TranslationDataset(path)
    # DataLoader 参数说明：
    #   batch_size   每次取多少条样本
    #   shuffle=True 每个 epoch 都打乱顺序（训练需要；测试一般不打乱）
    #   collate_fn   自定义"如何把多条样本打包成一个 batch"，这里用来 padding
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


if __name__ == '__main__':
    # 仅直接运行本文件时执行：自测 DataLoader 是否工作正常
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))   # 训练 batch 数 = 样本数 / batch_size
    print(len(test_dataloader))    # 测试 batch 数

    # 取第一个 batch 观察形状：for 循环迭代 DataLoader 会得到 (input, target) 的批次
    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor)  # input_tensor.shape: [batch_size, seq_len]
        print(target_tensor)  # target_tensor.shape : [batch_size, seq_len]（原注释误写为 [batch_size]，已校正）
        break
