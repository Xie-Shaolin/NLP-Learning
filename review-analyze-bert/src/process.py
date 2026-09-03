import pandas as pd
from datasets import load_dataset, ClassLabel
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from tokenizer import JiebaTokenizer

import config


def process():
    """
    数据预处理主流程：
    读取原始 CSV → 清洗数据 → 划分训练/测试集 → 分词编码 → 保存为处理后的数据集。
    """
    print('开始处理数据')

    # 1. 读取文件：使用 HuggingFace datasets 以 CSV 方式加载，返回 train split
    dataset = load_dataset('csv', data_files=str(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv'))['train']

    # 2. 过滤数据：删除无用的 'cat' 类别列，并去掉评论为空的样本
    dataset = dataset.remove_columns(['cat'])
    dataset = dataset.filter(lambda x: x['review'] is not None)

    # 3. 划分数据集：将 label 转为类别标签（0=negative, 1=positive），
    #    再按 8:2 分层切分，保证训练/测试集中正负样本比例一致
    # ClassLabel(...) —— 定义 label 列应该是什么类型
    # cast_column(...) —— 把这一列转换成这种类型
    dataset = dataset.cast_column('label', ClassLabel(names=['negative', 'positive']))
    print(dataset.features)
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
    print(dataset_dict)

    # 4. 创建分词器：加载本地预训练的 bert-base-chinese 分词器
    # tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
    # 4. 创建分词器：加载 网络上 预训练的 bert-base-chinese 分词器
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

    # 5. 构建数据集：定义批量编码函数，供 dataset.map 调用
    def batch_encode(batch):
        # 对评论批量分词编码，填充/截断到 SEQ_LEN 长度
        inputs = tokenizer(batch['review'], padding='max_length', truncation=True, max_length=config.SEQ_LEN)
        # 把标签一并放入 inputs，最终每样本包含 input_ids/attention_mask/token_type_ids/labels
        inputs['labels'] = batch['label']
        return inputs

    # 对训练/测试集统一执行编码，并移除原始文本列（后续只保留张量字段）
    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=['review', 'label'])

    # 6. 保存数据集到磁盘（按 split 存为 train/test 两个子目录）
    dataset_dict.save_to_disk(config.PROCESSED_DATA_DIR)

    print('数据处理完成')


if __name__ == '__main__':
    process()
