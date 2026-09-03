import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import get_dataloader
from tokenizer import JiebaTokenizer
import config
from model import ReviewAnalyzeModel


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个 epoch：遍历整个训练集，执行前向、反向与参数更新。
    :param model: 待训练的模型。
    :param dataloader: 训练集 DataLoader。
    :param loss_fn: 损失函数（BCEWithLogitsLoss）。
    :param optimizer: 优化器。
    :param device: 运行设备（cuda 或 cpu）。
    :return: 本 epoch 的平均损失。
    """
    # 累计本 epoch 所有 batch 的损失，用于最后求平均
    total_loss = 0
    # 切换到训练模式：启用 Dropout 等训练时行为
    model.train()
    for batch in tqdm(dataloader, desc='训练'):
        # 将 batch 中所有张量移动到设备
        inputs = {k: v.to(device) for k, v in batch.items()}
        # 取出标签并转为 float（BCEWithLogitsLoss 要求标签为浮点类型）
        labels = inputs.pop('labels').to(dtype=torch.float)

        # 前向传播得到 logits，shape: [batch_size]
        outputs = model(**inputs)
        # 计算损失
        loss = loss_fn(outputs, labels)

        # 反向传播计算梯度
        loss.backward()
        # 依据梯度更新一次参数
        optimizer.step()
        # 清空累积梯度，避免影响下一个 batch
        optimizer.zero_grad()

        # 累加当前 batch 的损失值
        total_loss += loss.item()
    # 返回本 epoch 平均损失
    return total_loss / len(dataloader)


def train():
    """训练主流程：准备资源、循环训练多个 epoch、记录日志并保存最优模型。"""
    # 1. 确定设备：有 GPU 则用 GPU，否则回退到 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 加载训练集 DataLoader
    dataloader = get_dataloader()
    # 3. 加载本地预训练 bert-base-chinese 分词器（这里没有用）
    # tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
    # 4. 实例化模型并移动到设备
    model = ReviewAnalyzeModel().to(device)
    # 5. 二分类损失：BCEWithLogitsLoss 内部先做 sigmoid 再算交叉熵，数值更稳定
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # 6. 优化器：Adam，学习率来自配置
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 7. 创建 TensorBoard 日志写入器，日志目录按当前时间戳命名，避免多次运行互相覆盖
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 记录历史最优损失，初始化为正无穷，保证第一个 epoch 一定能保存模型
    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        # 训练一个 epoch 并返回平均损失
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'Loss: {loss:.4f}')

        # 将当前 epoch 的损失记录到 TensorBoard
        writer.add_scalar('Loss', loss, epoch)

        # 若当前损失优于历史最优，则更新最优值并保存模型权重
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
            print('保存模型')

    # 训练结束，关闭日志写入器
    writer.close()


if __name__ == '__main__':
    train()
