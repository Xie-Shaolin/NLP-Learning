"""
训练模块
负责模型的训练流程：加载数据、定义模型/损失/优化器、逐轮训练并保存最优模型。
"""

import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import JiebaTokenizer
import config
from model import ReviewAnalyzeModel


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个 epoch（把训练集完整遍历一遍）

    :param model: 待训练的模型
    :param dataloader: 训练数据加载器
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 计算设备（cuda 或 cpu）
    :return: 本 epoch 的平均损失
    """
    # 累计本 epoch 所有 batch 的损失，最后求平均
    total_loss = 0
    # 切换到训练模式（会启用 Dropout 等训练时行为）
    model.train()
    # tqdm 用来显示训练进度条
    for inputs, targets in tqdm(dataloader, desc='训练'):
        # 把数据和标签搬到指定设备（GPU/CPU）
        inputs = inputs.to(device)  # inputs.shape: [batch_size, seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size]

        # 前向传播，得到模型预测分数
        outputs = model(inputs)
        # outputs.shape: [batch_size]

        # 计算损失
        loss = loss_fn(outputs, targets)

        # 反向传播：计算梯度
        loss.backward()
        # 根据梯度更新参数
        optimizer.step()
        # 清空梯度，避免累加到下一个 batch
        optimizer.zero_grad()

        # 累加当前 batch 的损失值（loss.item() 把张量转成 Python 数字）
        total_loss += loss.item()
    # 返回平均损失
    return total_loss / len(dataloader)


def train():
    """
    训练主流程：准备资源 -> 逐 epoch 训练 -> 保存最优模型
    """
    # 1. 选择设备：有 GPU 用 GPU，否则用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 加载训练数据
    dataloader = get_dataloader()
    # 3. 从词表文件加载分词器（用于获取词表大小和 pad 索引）
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    # 4. 构建模型并搬到设备上
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_index).to(device)
    # 5. 损失函数：BCEWithLogitsLoss 内部会自动做 sigmoid，再计算二分类交叉熵
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # 6. 优化器：Adam，学习率来自配置
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 7. TensorBoard 记录器：用当前时间作为日志目录，避免覆盖
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 初始最优损失设为正无穷，保证第一个 epoch 一定能保存模型
    best_loss = float('inf')
    # 循环训练 config.EPOCHS 轮
    for epoch in range(1, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        # 训练一个 epoch，返回平均损失
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'Loss: {loss:.4f}')

        # 把本 epoch 损失记录到 TensorBoard，便于可视化观察
        writer.add_scalar('Loss', loss, epoch)

        # 如果当前损失比历史最优还低，就保存模型权重
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
            print('保存模型')

    # 训练结束，关闭日志记录器
    writer.close()


if __name__ == '__main__':
    train()
