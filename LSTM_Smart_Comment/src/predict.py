"""
预测模块
加载训练好的模型，对输入的评论文本进行情感分析预测。
"""

import torch

import config
from model import ReviewAnalyzeModel
from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """
    批量预测（返回每个样本的正向概率）

    :param model: 已加载权重的模型
    :param inputs: 输入的词索引序列，shape: [batch_size, seq_len]
    :return: 预测结果列表，shape: [batch_size]，每个值是 0~1 之间的正向概率
    """
    # 切换到评估模式（关闭 Dropout 等训练时行为）
    model.eval()
    # 关闭梯度计算，预测时不需要反向传播，可节省显存并加速
    with torch.no_grad():
        # 前向传播得到预测分数（logits）
        output = model(inputs)
        # output.shape: [batch_size]
    # 用 sigmoid 把分数映射到 0~1 之间，作为正向概率
    batch_result = torch.sigmoid(output)
    # 转成 Python 列表返回
    return batch_result.tolist()


def predict(text, model, tokenizer, device):
    """
    对单条文本做情感预测

    :param text: 输入的评论文本
    :param model: 模型
    :param tokenizer: 分词器（用于把文本编码成索引）
    :param device: 计算设备
    :return: 正向概率（0~1 之间的浮点数）
    """
    # 1. 处理输入：分词并把文本编码成固定长度索引序列
    indexes = tokenizer.encode(text, seq_len=config.SEQ_LEN)
    # 转成张量，并增加 batch 维度 [1, seq_len]
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    # 搬到指定设备
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [batch_size, seq_len]

    # 2. 调用批量预测，得到 batch 结果
    batch_result = predict_batch(model, input_tensor)

    # 3. 只返回第一条（因为只输入了一条文本）
    return batch_result[0]


def run_predict():
    """
    交互式预测入口：循环读取用户输入，输出情感分析结果
    """
    # ---- 准备资源 ----
    # 1. 确定计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载词表并构建分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")

    # 3. 构建模型结构并加载训练好的权重
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    print("欢迎情感分析模型(输入q或者quit退出)")

    # ---- 交互循环 ----
    while True:
        user_input = input("> ")
        # 输入 q 或 quit 时退出
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        # 输入为空（或只有空格）时提示重新输入
        if user_input.strip() == '':
            print("请输入内容")
            continue

        # 对用户输入做预测，得到正向概率
        result = predict(user_input, model, tokenizer, device)
        # 概率 > 0.5 判定为正向，否则负向
        if result > 0.5:
            print(f"正向（置信度：{result}）")
        else:
            print(f"负向（置信度：{1 - result}）")


if __name__ == '__main__':
    run_predict()
