"""
模型评估模块
在测试集上评估模型性能，计算 Top-1 和 Top-5 准确率
"""

import torch
import config
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    """
    在测试集上评估模型，计算 Top-1 和 Top-5 准确率
    
    准确率定义：
    - Top-1 准确率：预测结果中排名第一的词与真实目标词相等的比例
    - Top-5 准确率：真实目标词出现在预测结果的前 5 个候选词中的比例
    
    Args:
        model (InputMethodModel): 已训练好的输入法模型
        test_dataloader (DataLoader): 测试集数据加载器
        device (torch.device): 计算设备（CPU 或 GPU）
    
    Returns:
        tuple: (top1_acc, top5_acc)
            - top1_acc (float): Top-1 准确率，范围 [0, 1]
            - top5_acc (float): Top-5 准确率，范围 [0, 1]
    """
    # Top-1 正确样本计数器
    top1_acc_count = 0
    # Top-5 正确样本计数器
    top5_acc_count = 0
    # 总样本计数器
    total_count = 0
    
    # 遍历测试集的每个批次
    for inputs, targets in test_dataloader:
        # 将输入张量移动到指定计算设备
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        
        # 将目标张量转换为 Python 列表，便于逐元素比较
        targets = targets.tolist()
        # targets.shape: [batch_size] 例如 [1, 3, 5]
        
        # 调用批量预测函数，获取每个样本的 Top-5 预测词索引
        top5_indexes_list = predict_batch(model, inputs)
        # top5_indexes_list.shape: [batch_size, 5] 例如 [[1,3,5,7,8],[1,3,5,7,8],...]
        
        # 逐样本比较预测结果与真实标签
        # zip() 是 Python 的 内置函数 ，作用是： 把多个可迭代对象"按位置"打包成元组，每个元组包含来自每个可迭代对象的对应元素。
        for target, top5_indexes in zip(targets, top5_indexes_list):
            # 总样本数加 1
            total_count += 1
            # 判断 Top-1 是否正确：预测的第一个词是否等于真实目标
            if target == top5_indexes[0]:
                top1_acc_count += 1
            # 判断 Top-5 是否正确：真实目标是否出现在前 5 个预测中
            if target in top5_indexes:
                top5_acc_count += 1
    
    # 计算准确率：正确样本数 / 总样本数
    return top1_acc_count / total_count, top5_acc_count / total_count


def run_evaluate():
    """
    模型评估主流程
    
    执行步骤：
    1. 确定计算设备（GPU/CPU）
    2. 加载分词器（词表）
    3. 初始化模型并加载预训练权重
    4. 加载测试集数据加载器
    5. 调用 evaluate 函数计算准确率并打印结果
    """
    # 准备资源阶段
    
    # Step 1: 确定计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 2: 加载分词器（词表）
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")

    # Step 3: 初始化模型并加载预训练权重
    # 根据词表大小创建模型实例，并移动到指定设备
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    # 从文件加载训练好的最佳模型权重（best.pth）
    # torch.load：从磁盘读数据
    # model.load_state_dict：把数据填入模型
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))
    print("模型加载成功")

    # Step 4: 加载测试集数据加载器
    # train=False 表示加载测试集
    test_dataloader = get_dataloader(train=False)

    # Step 5: 执行评估逻辑，计算准确率
    top1_acc, top5_acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"top1_acc: {top1_acc}")
    print(f"top5_acc: {top5_acc}")


if __name__ == '__main__':
    """
    主函数：执行模型评估流程
    直接运行此文件可在测试集上评估模型性能
    """
    run_evaluate()
