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
        # inputs：源序列（中文），不包含 <sos> 和 <eos>，但是包含 <pad>
        # targets：目标序列（英文），包含 <pad>, <sos>, <eos>
        encoder_inputs = inputs.to(device)  # inputs.shape: [batch_size, src_seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size, tgt_seq_len]

        # 错位切分：解码器"第 i 步"预测的是目标序列第 i+1 个词。
        # decoder_inputs   = 去掉最后一个 eos 的序列（作为输入）
        # decoder_targets  = 去掉最前面 sos 的序列（作为要预测的标准答案）
        # [:, :-1] ： 取batch_size行的所有元素，从第0列开始，到倒数第二个列（-1）结束。-1 表示最后一个数（不包含）
        # [:, 1:] ： 取batch_size行的所有元素，从第1列开始，到最后
        decoder_inputs = targets[:, :-1]  # decoder_inputs.shape: [batch_size, seq_len]
        decoder_targets = targets[:, 1:]  # decoder_targets.shape: [batch_size, seq_len]
        '''
            targets = [[sos, I,   love, you, eos],
                        [sos, He,  is,   a,   eos]]     # shape [2, 5]
            decoder_inputs = targets[:, :-1]  
                    →  [[sos, I,   love, you],        ← 去掉末尾 eos → 解码器输入
                        [sos, He,  is,   a  ]]

            decoder_targets = targets[:, 1:]   
                    →  [[I,   love, you, eos],        ← 去掉开头 sos → 标准答案
                        [He,  is,   a,   eos ]]
            1. <sos> 只做输入，不做预测</sos> ：序列生成的起点是 <sos> ，它本身不是需要预测的内容，所以答案里要把它去掉（ [:, 1:] ）。
            2. <eos> 只做预测，不做输入</eos> ： <eos> 是终点标记，没有下一步，所以输入里要把它去掉（ [:, :-1] ）。
            3. 教师强制 ：整个目标序列是真实答案，直接整体喂给解码器（而不是用模型上一步的预测输出），让每一步都"看着标准答案"学习，收敛更稳定。
            4. 如果句子是 [sos, I, love, you, eos, pad, pad]，那么decoder_inputs = targets[:, :-1]  会是 [sos, I, love, you, eos, pad]，eos还在
            5. 但是由于 loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)，
                而当 decoder_inputs 为 eos 时，decoder_targets为 pad，所以不会计算 loss。
        '''

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
            # input：sos, I,   love, you
            # 取出第 i 步的输入词，unsqueeze(1) 增加时间步维 -> [batch_size, 1]
            # [:, i] ： 取batch_size行的所有元素，第i列。可以理解为，对同一个batch，拿出同一个时间步 i 的输入词去做运算
            decoder_input = decoder_inputs[:, i].unsqueeze(1)  # decoder_input.shape: [batch_size, 1]
            # 单步解码：输出该步预测分布 + 更新隐藏状态（作为下一步的输入）
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            # decoder_output.shape: [batch_size, 1, vocab_size]
            decoder_outputs.append(decoder_output)

        # 把列表里所有时间步的输出拼成一个大张量
        # torch.cat(列表, dim=1)：沿时间步维拼接
        # `cat` 是 PyTorch 中用于**沿某一个维度把多个 tensor 拼接起来**的操作。
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs.shape: [batch_size, seq_len, vocab_size]

        # reshape(-1, ...)：-1 表示自动推断该维大小，把三维展平成二维
        # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        # decoder_outputs.shape 得到一个元组，包含3 个元素：(batch_size, seq_len, vocab_size)
        # decoder_outputs.shape[-1] 得到 vocab_size，-1表示最后一个元素
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        # decoder_outputs.shape: [batch_size * seq_len, vocab_size]

        # 标准答案同样展平：[batch_size, seq_len] -> [batch_size * seq_len]
        decoder_targets = decoder_targets.reshape(-1)
        '''
            为什么要展平？
            展平的 根本原因是 PyTorch 的 CrossEntropyLoss 的接口要求 ——它只接受二维输入，不接受三维。
            PyTorch 默认的交叉熵损失只接受两种形状：
            - input（预测 logits） ： (N, C) —— N 是样本总数，C 是类别数（这里就是 vocab_size ）
            - target（标准答案） ： (N,) —— N 个整数索引，每个位置是真实词的词表索引
        '''


        # 交叉熵损失：逐位置对比预测分布与真实词索引。
        # ignore_index 已在外面配置为 <pad> 索引，padding 位置不计入损失。
        # decoder_outputs.shape: [batch_size * seq_len, vocab_size]
        # decoder_targets.shape: [batch_size * seq_len]
        # 因为 训练过程中，每一个时间步都会产生一个损失值，该样本的总损失，就是所有时间步的损失值逐步累加的结果。
        # 多分类交叉熵损失函数的计算公式为：L = 对 batch 中每个样本的交叉熵 loss 求平均
        #                                = - mean( sum(y * log(y_hat), dim=类别维度) )
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
    
    '''
        len(dataloader) 是这个 epoch 里一共会迭代多少个 batch （批次数）
        --> len(dataloader) 调用 dataloader 的 __len__ 方法
        --> 接着 batch_sampler.__len__() ：批次数 len(dataloader) = ceil(N / batch_size)
        --> sampler.__len__() :  样本总数 N（默认 SequentialSampler/RandomSampler）
        --> dataset.__len__()   样本总数 N (= len(self.data)) (这个是自己写的Dataset的__len__方法)
    '''
    # 所有 batch 损失的平均值 = 总损失 / 这个 epoch 里一共会迭代多少个 batch （批次数）
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
    model = TranslationModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, 
                            zh_tokenizer.pad_token_index, en_tokenizer.pad_token_index).to(device)

    '''
        5. 损失函数
        loss = loss_fn(decoder_outputs, decoder_targets)
        CrossEntropyLoss：多分类交叉熵，内部自带 softmax；
        ignore_index 让 <pad> 位置不参与 loss 计算，避免 padding 干扰训练。
        举例说明：
            - 假设词表里： <pad>=0, <sos>=1, <eos>=2, I=3, love=4, you=5, he=6, is=7, a=8 。
            - 一个 batch 有 2 条英文句子，第 2 句短，被 pad_sequence 补到与第 1 句等长：
                targets 原始（形状 [2, 6]）:
                    句子1: [<sos>, I,     love, you, <eos>, <pad>]  → [1, 3, 4, 5, 2, 0]
                    句子2: [<sos>, he,    is,   <eos>, <pad>, <pad>] → [1, 6, 7, 2, 0, 0]
            - targets[:, 1:] ， 去掉头部 <sos> 后再看 pad ：
                decoder_inputs  = targets[:, :-1]   →  [2, 5]
                    句子1: [1, 3, 4, 5, 2]
                    句子2: [1, 6, 7, 2, 0]
                decoder_targets = targets[:, 1:]    →  [2, 5]
                    句子1: [3, 4, 5, 2, 0]
                    句子2: [6, 7, 2, 0, 0]
        经过 decoder_targets.reshape(-1) 展平成一行：
            decoder_targets: [3, 4, 5, 2, 0, 6, 7, 2, 0, 0]
            decoder_outputs 是 [N, vocab_size] （N=10），每行是一步的 logits，
                    其中最后一个位置（下标 9，对应 <pad> ）的 logits 随意，反正要被忽略。
        如果不设 ignore_index ， CrossEntropyLoss 会对 所有 10 个位置 都算损失：
            loss = -1/10 × [ log p(3) + log p(4) + ... + log p(0) + log p(0) ]
                        ↑ 全部都被算进去了，包括 pad
            问题：最后那个位置的标准答案是 <pad>=0 ，但它是 人为补出来的假数据 ，没有任何真实语义。
            如果让模型去"学"在某个 <eos> 之后预测 <pad> ，等于教它"句子结束后还要输出 [pad] "。
            这不仅让 padding 位置干扰梯度，还会拉低 model 对 </s 结束> 的学习。
            同时分母是 10，真实有效的预测只有 9 个，均摊也被稀释了。
        设了 ignore_index=0 后：标准答案等于 0 的那些位置，直接从损失里剔除 
                —— 既不算进求和 sum，也不计入分母 count。
            decoder_targets = [3, 4, 5, 2, 0,  6, 7, 2, 0, 0]
                └── 句子1: 4个有效 + 1个pad ──┘└─ 句子2: 3个有效 + 2个pad ─┘
            loss = -1/7 × [ log p(3) + log p(4) + log p(5) + log p(2)    ← 句子1
                            + log p(6) + log p(7) + log p(2) ]           ← 句子2
    '''
    # ignore_index 会让 CrossEntropyLoss 在计算时，把所有 标签值（ target ）等于该值的元素位置 从损失求和与平均中剔除，
    # 即不对这些位置计算和反向传播梯度。
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
