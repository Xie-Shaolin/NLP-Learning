"""
模型训练模块
实现 RNN 输入法模型的完整训练流程，包括：
- 单轮次训练（train_one_epoch）
- 多轮次训练主循环（train）
- TensorBoard 可视化日志记录
- 最佳模型自动保存
"""

import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from model import InputMethodModel
import config
from tokenizer import JiebaTokenizer


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个完整的 epoch（遍历一次训练集）
    
    Args:
        model (InputMethodModel): 待训练的输入法模型
        dataloader (DataLoader): 训练集数据加载器
        loss_fn (nn.Module): 损失函数，此处为交叉熵损失（CrossEntropyLoss）
        optimizer (torch.optim.Optimizer): 优化器，此处为 Adam
        device (torch.device): 计算设备（CPU 或 GPU）
    
    Returns:
        float: 当前 epoch 的平均损失值（每个批次的平均损失）
    """
    # 将模型设置为训练模式
    # 影响 Dropout、BatchNorm 等层的行为（训练时启用 Dropout）
    # 设置 training=True 后，这些层会根据模型状态（如梯度）进行更新，而不是固定值。
    model.train()
    
    # 累计损失值，用于计算平均损失
    total_loss = 0
    
    # 遍历训练集的每个批次（batch）
    # tqdm 显示训练进度条，desc 设置进度条文字
    for inputs, targets in tqdm(dataloader, desc='训练'):
        # 将输入和目标张量移动到指定计算设备（CPU/GPU）
        inputs = inputs.to(device)
        targets = targets.to(device)
        # inputs.shape: [batch_size, seq_len]
        # targets.shape: [batch_size]

        # ===== 前向传播 =====
        # 模型根据输入序列预测每个词的分数（logits）
        # model(inputs)：触发 forward 方法
        outputs = model(inputs)
        # outputs.shape: [batch_size, vocab_size]
        
        # 计算损失：预测输出与真实目标之间的交叉熵
        # CrossEntropyLoss 内部会自动对 outputs 做 Softmax，然后计算负对数似然
        loss = loss_fn(outputs, targets)

        # ===== 反向传播与参数更新 =====
        # 反向传播：根据损失计算每个参数的梯度（∂loss/∂w）
        loss.backward()
        
        # 使用优化器更新模型参数：w = w - lr * ∇w
        optimizer.step()
        
        # 清空梯度缓存，防止梯度累积
        # PyTorch 默认会累积梯度，所以每次更新后必须手动清零
        # zero_grad() 方法会将所有参数的梯度设为 0，准备下一次反向传播
        # 这是确保梯度累加不会影响模型训练的重要一步
        # 否则，梯度会继续累加，导致模型参数更新过快或过慢
        optimizer.zero_grad()

        # 将当前批次的损失值（Python 标量）累加到总损失中
        # .item() 从单元素张量中取出 Python float 值
        total_loss += loss.item()
    
    # 返回平均损失：总损失 / 批次数
    return total_loss / len(dataloader)


def train():
    """
    模型训练主流程
    
    执行步骤：
    1. 确定计算设备（GPU/CPU）
    2. 加载训练数据
    3. 加载分词器（词表）
    4. 初始化模型
    5. 定义损失函数（交叉熵）
    6. 定义优化器（Adam）
    7. 初始化 TensorBoard 日志记录器
    8. 多轮次训练循环：
       - 每个 epoch 调用 train_one_epoch
       - 记录损失到 TensorBoard
       - 保存最佳模型（损失最低的权重）
    9. 关闭日志记录器
    """
    # Step 1: 确定计算设备
    # 优先使用 GPU（cuda），GPU 不存在则使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 2: 加载训练数据
    # 获取训练集 DataLoader
    dataloader = get_dataloader()

    # Step 3: 加载分词器（词表）
    # 从保存的词表文件创建 JiebaTokenizer 实例
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')

    # Step 4: 初始化模型
    # 词表大小决定模型 Embedding 层和输出层的维度
    # .to(device) 将模型参数移动到指定设备
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)

    # Step 5: 定义损失函数
    # CrossEntropyLoss：多分类任务的标准损失函数
    # 内部包含 Softmax + NLLLoss（负对数似然损失）
    loss_fn = torch.nn.CrossEntropyLoss()

    # Step 6: 定义优化器
    # Adam 优化器：自适应学习率的随机梯度下降变种，收敛快、效果好
    # model.parameters(): 模型中所有可训练的参数
    # lr: 学习率，控制参数更新的步长
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Step 7: 初始化 TensorBoard 日志记录器
    # log_dir: 日志保存目录，使用时间戳命名避免覆盖
    # time.strftime 格式化当前时间为 "年-月-日_时-分-秒" 格式
    # SummaryWriter(log_dir=某个路径) 表示将日志保存到该路径下
    # SummaryWriter 是 PyTorch 自带的 TensorBoard 日志记录器 ，
    # 作用：把训练过程中的 loss 、准确率、模型图等数据写入磁盘，方便用 TensorBoard 可视化查看训练曲线。
    #  打开 tensorboard --logdir=/Users/xieshaolin/workpalce/RNN_Smart_Input/logs
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # ===== 开始多轮训练 =====
    # 初始化最佳损失为无穷大，任何真实损失都会比它小
    best_loss = float('inf')
    
    # 循环训练 config.EPOCHS 轮
    for epoch in range(1, 1 + config.EPOCHS):
        # 打印分隔线和当前轮次信息
        print("=" * 10, f" Epoch: {epoch} ", "=" * 10)
        
        # 训练一个完整的 epoch，返回平均损失
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"loss:{loss}")

        # 将损失记录到 TensorBoard
        # 标量名 'loss'，值为 loss，横轴为 epoch 编号
        writer.add_scalar('loss', loss, epoch)

        # 保存最佳模型：如果当前损失优于历史最佳
        if loss < best_loss:
            # 更新最佳损失记录
            best_loss = loss
            # 保存模型权重（state_dict 只保存参数，不保存模型结构）
            # best.pth 始终存储训练过程中损失最低的模型权重
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pth')
            print("模型保存成功")

    # 训练结束，关闭 TensorBoard 日志记录器
    writer.close()


if __name__ == '__main__':
    """
    主函数：启动模型训练
    直接运行此文件即可开始训练模型
    """
    train()
