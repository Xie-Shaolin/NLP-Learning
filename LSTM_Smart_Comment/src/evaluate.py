"""
评估模块
加载模型和测试集，计算模型在测试集上的准确率（accuracy）。
"""

import torch
import config
from model import ReviewAnalyzeModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    """
    在测试集上评估模型，计算准确率

    :param model: 已加载权重的模型
    :param test_dataloader: 测试数据加载器
    :param device: 计算设备
    :return: 准确率（0~1 之间的浮点数）
    """
    # 统计样本总数和预测正确的数量
    total_count = 0
    correct_count = 0
    for inputs, targets in test_dataloader:
        # 输入搬到设备上
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        # 真实标签转成 Python 列表，方便逐个比较
        targets = targets.tolist()
        # targets.shape: [batch_size] e.g.[0,1,0,1]

        # 批量预测，得到正向概率列表
        batch_result = predict_batch(model, inputs)
        # batch_result.shape: [batch_size] e.g. [0.1, 0.2, 0.9, 0.3]

        # 逐个样本比较预测结果与真实标签
        for result, target in zip(batch_result, targets):
            # 概率 > 0.5 判定为正类(1)，否则为负类(0)
            result = 1 if result > 0.5 else 0
            # 预测正确则计数 +1
            if result == target:
                correct_count += 1
            # 样本总数 +1
            total_count += 1
    # 返回准确率 = 正确数 / 总数
    return correct_count / total_count


def run_evaluate():
    """
    评估入口：加载资源并在测试集上评估模型
    """
    # ---- 准备资源 ----
    # 1. 确定计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载词表并构建分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")

    # 3. 构建模型结构并加载训练好的权重
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    # 4. 加载测试集（train=False 表示测试集）
    test_dataloader = get_dataloader(train=False)

    # 5. 评估并打印准确率
    acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"acc: {acc}")


if __name__ == '__main__':
    run_evaluate()
