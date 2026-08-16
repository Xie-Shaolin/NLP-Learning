"""
模型预测模块
提供批量预测和单条文本预测功能，并包含交互式预测演示程序
"""

import jieba
import torch
import config
from model import InputMethodModel
from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """
    批量预测函数：对一批输入序列同时预测下一个词的 Top-5 候选
    
    Args:
        model (InputMethodModel): 已训练好的输入法模型
        inputs (torch.Tensor): 输入张量，形状为 [batch_size, seq_len]
                            batch_size: 批次大小（一次预测的样本数）
                            seq_len: 序列长度（历史输入词数）
    
    Returns:
        list: 二维列表，形状为 [batch_size, 5]
            每个元素是对应样本预测概率最高的 5 个词的索引列表
            按预测概率从高到低排序
    """
    # 将模型设置为评估模式
    # 影响 Dropout、BatchNorm 等层的行为（评估时不使用 Dropout）
    model.eval()
    
    # 禁用梯度计算，减少内存开销并加速推理
    # 预测阶段不需要反向传播，因此不需要计算梯度
    # torch.no_grad() 是一个上下文管理器，用于关闭梯度计算，with的代码运行完毕后会自动开启梯度计算
    with torch.no_grad():
        # 前向传播：模型输出每个词的预测分数（logits）
        output = model(inputs)
        # output.shape: [batch_size, vocab_size]
    
    # 使用 torch.topk 选取分数（logits）最高的 k=5 个词的索引
    # .indices 只取索引部分，忽略具体的分数值
    top5_indexes = torch.topk(output, k=5).indices
    # top5_indexes.shape: [batch_size, 5]

    # 将 PyTorch 张量转换为 Python 列表，便于后续处理
    top5_indexes_list = top5_indexes.tolist()
    return top5_indexes_list


def predict(text, model, tokenizer, device):
    """
    单条文本预测：输入一段文本，预测下一个词的 Top-5 候选词
    
    Args:
        text (str): 用户输入的历史文本（将作为上下文）
        model (InputMethodModel): 已训练好的输入法模型
        tokenizer (JiebaTokenizer): 分词器，用于文本编码和解码
        device (torch.device): 计算设备（CPU 或 GPU）
    
    Returns:
        list: Top-5 预测词的字符串列表，按预测概率从高到低排序
            例如：['你好', '天气', '今天', '不错', '的']
    """
    # Step 1: 处理输入文本
    # 对文本进行分词并编码为词索引序列
    indexes = tokenizer.encode(text)
    # 将一维索引列表包装为二维张量，增加 batch 维度（batch_size=1）
    # [seq_len] -> [1, seq_len]
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    # 将张量移动到指定设备（CPU/GPU）
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [batch_size, seq_len]

    # Step 2: 执行预测逻辑
    # 调用批量预测函数（batch_size=1）
    top5_indexes_list = predict_batch(model, input_tensor)
    # top5_indexes_list.shape: [1, 5]，也就是只有一个样本
    # 取出第一个（也是唯一一个）样本的 Top-5 索引
    # 根据索引从词表中还原出对应的词字符串
    top5_tokens = [tokenizer.index2word[index] for index in top5_indexes_list[0]]
    return top5_tokens


def run_predict():
    """
    交互式预测程序
    
    功能：
    1. 加载分词器和预训练模型
    2. 进入命令行交互模式
    3. 用户持续输入文字，程序根据历史输入预测下一个词的 Top-5
    4. 输入 'q' 或 'quit' 退出程序
    """
    # 准备资源阶段
    
    # Step 1: 确定计算设备
    # 优先使用 GPU（cuda），如果不可用则回退到 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 2: 加载词表 / 分词器
    # 从已保存的词表文件中加载 JiebaTokenizer 实例
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")

    # Step 3: 加载模型
    # 初始化模型实例，词表大小由分词器提供
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    # 加载预训练好的模型权重文件（best.pth 是训练过程中保存的最佳模型）
    # load_state_dict 将权重加载到模型中
    # torch.load：从磁盘读数据
    # model.load_state_dict：把数据填入模型
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))
    print("模型加载成功")

    # 进入交互模式
    print("欢迎使用输入法模型(输入q或者quit退出)")
    
    # 保存用户的完整输入历史，作为预测的上下文
    input_history = ''
    while True:
        # 等待用户输入
        user_input = input("> ")
        
        # 检查退出命令
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        
        # 检查空输入
        if user_input.strip() == '':
            print("请输入内容")
            continue
        
        # 将新输入追加到历史记录中，保持上下文完整
        input_history += user_input
        print(f'输入历史:{input_history}')
        
        # 基于完整历史输入预测下一个词的 Top-5
        top5_tokens = predict(input_history, model, tokenizer, device)
        print(f'预测结果:{top5_tokens}')


if __name__ == '__main__':
    """
    主函数：启动交互式预测程序
    直接运行此文件可进入命令行预测界面
    """
    run_predict()
