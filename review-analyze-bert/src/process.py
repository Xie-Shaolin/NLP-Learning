import pandas as pd
from datasets import load_dataset, ClassLabel
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from tokenizer import JiebaTokenizer

import config


def process():
    print('开始处理数据')
    # 读取文件
    dataset = load_dataset('csv',data_files=str(config.RAW_DATA_DIR/'online_shopping_10_cats.csv'))['train']

    # 过滤数据
    dataset = dataset.remove_columns(['cat'])
    dataset = dataset.filter(lambda x: x['review'] is not None)

    # 划分数据集
    dataset = dataset.cast_column('label',ClassLabel(names=['negative','positive']))
    dataset_dict = dataset.train_test_split(test_size=0.2,stratify_by_column='label')
    print(dataset_dict)

    # 创建Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR/'bert-base-chinese')

    # 构建数据集
    def batch_encode(batch):
        inputs = tokenizer(batch['review'], padding='max_length', truncation=True, max_length=config.SEQ_LEN)
        inputs['labels'] = batch['label']
        return inputs

    dataset_dict = dataset_dict.map(batch_encode, batched=True,remove_columns=['review','label'])

    # 保存数据集
    dataset_dict.save_to_disk(config.PROCESSED_DATA_DIR)

    print('数据处理完成')


if __name__ == '__main__':
    process()
