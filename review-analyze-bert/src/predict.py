import jieba
import torch
from transformers import AutoTokenizer

import config
from model import ReviewAnalyzeModel
from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """
    对一个批次的数据执行预测，返回每个样本的预测概率。
    :param model: 已加载权重的 ReviewAnalyzeModel 模型。
    :param inputs: 分词/编码后的输入字典，shape: [batch_size, seq_len]。
    :return: 每个样本的预测概率列表，取值范围 [0,1]，shape: [batch_size]。
    """
    # 切换到评估模式：关闭 Dropout、BatchNorm 等训练时行为
    model.eval()
    # 关闭梯度计算：预测阶段无需反向传播，可节省显存并加速
    with torch.no_grad():
        # 前向传播得到 logits，shape: [batch_size]
        output = model(**inputs)
    # sigmoid 将 logits 映射到 [0,1] 的概率，表示“正向”的置信度
    batch_result = torch.sigmoid(output)
    return batch_result.tolist()


def predict(text, model, tokenizer, device):
    """
    对单条文本进行情感预测。
    :param text: 待预测的原始文本字符串。
    :param model: 已加载权重的模型。
    :param tokenizer: HuggingFace 分词器。
    :param device: 运行设备（cuda 或 cpu）。
    :return: 单个标量概率，表示该文本为“正向”的置信度。
    """
    # 1. 处理输入：分词并编码成模型所需格式
    #    padding='max_length'：不足 SEQ_LEN 时用 pad 补齐；
    #    truncation=True：超出 SEQ_LEN 时截断；
    #    return_tensors='pt'：返回 PyTorch 张量。
    inputs = tokenizer(text, padding='max_length', truncation=True,
                       max_length=config.SEQ_LEN, return_tensors='pt')

    # 2. 将输入张量移动到指定设备（GPU/CPU），与模型保持一致
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 调用批量预测，取第 0 个（因为只有一个样本）结果
    batch_result = predict_batch(model, inputs)

    return batch_result[0]


def run_predict():
    """启动交互式命令行预测：循环读取用户输入并输出情感判断。"""
    # 准备资源
    # 1. 确定设备：有 GPU 则用 GPU，否则回退到 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载与训练/处理阶段一致的分词器
    # tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    # 3. 加载模型结构并载入训练好的权重
    model = ReviewAnalyzeModel().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    print("欢迎情感分析模型(输入q或者quit退出)")

    # 循环读取输入，直到用户输入 q/quit 退出
    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        # 空输入（仅空格）时提示并继续等待，避免送入无效文本
        if user_input.strip() == '':
            print("请输入内容")
            continue

        # 预测单条文本，得到“正向”的置信度
        # p > 0.5  →  预测为 1（正类）
        # p ≤ 0.5  →  预测为 0（负类）
        result = predict(user_input, model, tokenizer, device)
        # 置信度 > 0.5 判定为正向，否则为负向
        if result > 0.5:
            print(f"正向（置信度：{result}）")
        else:
            print(f"负向（置信度：{1 - result}）")


if __name__ == '__main__':
    run_predict()
