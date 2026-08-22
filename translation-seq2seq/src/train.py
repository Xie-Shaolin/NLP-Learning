# ========== 训练模块 ==========
# 负责整个训练流程：加载数据/词表/模型，按 epoch 迭代训练，
# 用 TensorBoard 记录 loss，并保存验证集上最优的模型权重。
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """训练一个 epoch：遍历整个数据集一次，返回该 epoch 的平均损失。

    核心是"教师强制（teacher forcing）"训练：把真实目标序列整体喂给解码器，
    而不是用模型自己上一步的输出，这样训练更稳定、收敛更快。

    :param model: TranslationModel 实例
    :param dataloader: 训练数据加载器
    :param loss_fn: 损失函数（交叉熵）
    :param optimizer: 优化器（Adam）
    :param device: 计算设备（cuda/cpu）
    :return: 本 epoch 的平均损失（float）
    """
    total_loss = 0
    model.train()  # 切换到训练模式（影响 Dropout/BatchNorm 等层的行为）
    for inputs, targets in tqdm(dataloader, desc='训练'):
        encoder_inputs = inputs.to(device)  # inputs.shape: [batch_size, src_seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size, tgt_seq_len]

        # 错位切分：解码器"第 i 步"预测的是目标序列第 i+1 个词。
        # decoder_inputs   = 去掉最后一个 eos 的序列（作为输入）
        # decoder_targets  = 去掉最前面 sos 的序列（作为要预测的标准答案）
        decoder_inputs = targets[:, :-1]  # decoder_inputs.shape: [batch_size, seq_len]
        decoder_targets = targets[:, 1:]  # decoder_targets.shape: [batch_size, seq_len]

        # 前向传播
        # 编码阶段：中文句子 -> 语义向量（context vector）
        context_vector = model.encoder(encoder_inputs)
        # context_vector.shape: [batch_size, hidden_size]

        # 解码阶段
        # unsqueeze(0)：在第 0 维插入一个维度，把 [batch_size, hidden_size]
        # 变成 [1, batch_size, hidden_size]，以匹配 GRU 要求的 (层数, batch, hidden) 格式。
        decoder_hidden = context_vector.unsqueeze(0)
        # decoder_hidden_0.shape: [1, batch_size, hidden_size]

        decoder_outputs = []  # 缓存每一步的解码输出

        seq_len = decoder_inputs.shape[1]  # 解码步数 = 目标序列长度
        for i in range(seq_len):
            # 取出第 i 步的输入词，unsqueeze(1) 增加时间步维 -> [batch_size, 1]
            decoder_input = decoder_inputs[:, i].unsqueeze(1)  # decoder_input.shape: [batch_size, 1]
            # 单步解码：输出该步预测分布 + 更新隐藏状态（作为下一步的输入）
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            # decoder_output.shape: [batch_size, 1, vocab_size]
            decoder_outputs.append(decoder_output)

        # 把列表里所有时间步的输出拼成一个大张量
        # torch.cat(列表, dim=1)：沿时间步维拼接
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs.shape: [batch_size, seq_len, vocab_size]

        # reshape(-1, ...)：-1 表示自动推断该维大小，把三维展平成二维
        # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        # decoder_outputs.shape: [batch_size * seq_len, vocab_size]

        # 标准答案同样展平：[batch_size, seq_len] -> [batch_size * seq_len]
        decoder_targets = decoder_targets.reshape(-1)

        # 交叉熵损失：逐位置对比预测分布与真实词索引。
        # ignore_index 已在外面配置为 <pad> 索引，padding 位置不计入损失。
        loss = loss_fn(decoder_outputs, decoder_targets)

        # 反向传播（PyTorch 三步曲，顺序有讲究）：
        # 1. loss.backward()    根据损失自动求每个参数的梯度
        # 2. optimizer.step()   用梯度更新参数
        # 3. optimizer.zero_grad() 清空旧梯度，防止跨 batch 累加
        #    （放在 step 之后是为了本步刚用完梯度就立刻清零；顺序与经典写法一致）
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # loss.item() 把张量中的数值取出为 Python float；累加所有 batch 的损失
        total_loss += loss.item()

    # 平均损失 = 总损失 / batch 数（len(dataloader) 即批次数）
    return total_loss / len(dataloader)


def train():
    """训练主流程：准备资源 -> 循环训练 -> TensorBoard 记录 -> 保存最优模型。"""
    # 1. 设备
    # torch.cuda.is_available() 判断是否有可用 GPU；'cuda' 有则用 GPU，否则退回 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 数据
    dataloader = get_dataloader()

    # 3. 分词器
    # 从之前 process.py 保存的词表文件加载分词器，用来获取词表大小与特殊 token 索引
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')

    # 4. 模型
    # 用中英词表大小和各自的 <pad> 索引构造模型，并 .to(device) 迁移到计算设备
    model = TranslationModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index).to(device)

    # 5. 损失函数
    # CrossEntropyLoss：多分类交叉熵，内部自带 softmax；
    # ignore_index 让 <pad> 位置不参与 loss 计算，避免 padding 干扰训练。
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)

    # 6. 优化器
    # Adam：自适应学习率的优化器；model.parameters() 提供所有可训练参数；
    # lr 是学习率。
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 7. TensorBoard Writer
    # SummaryWriter 把标量/图写入日志目录；time.strftime 用时间戳做子目录名，区分每次运行。
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')  # float('inf') 正无穷，保证第一个 epoch 一定能触发保存
    for epoch in range(1, config.EPOCHS + 1):
        # range(1, EPOCHS+1)：生成 1..EPOCHS，让显示从 1 开始更友好
        print(f'========== Epoch {epoch} ==========')
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        # f-string 格式化输出，{loss:.4f} 保留 4 位小数
        print(f'Loss: {loss:.4f}')

        # 记录到Tensorboard
        # add_scalar(tag, value, step)：画一条以 epoch 为横轴的损失曲线
        writer.add_scalar('Loss', loss, epoch)

        # 保存模型
        # 只保留历史最优（loss 更小才覆盖保存），防止过拟合后模型退化
        if loss < best_loss:
            best_loss = loss
            # model.state_dict() 返回所有参数/缓冲区字典；torch.save 序列化到 .pt 文件
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
            print('保存模型')

    writer.close()  # 训练结束关闭 writer，把缓存数据写入磁盘


if __name__ == '__main__':
    # 仅直接运行本文件时启动训练
    train()
