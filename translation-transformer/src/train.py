import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc='训练'):
        encoder_inputs = inputs.to(device)  # inputs.shape: [batch_size, src_seq_len]
        targets = targets.to(device)  # targets.shape: [batch_size, tgt_seq_len]

        # decoder_inputs   = 去掉最后一个 eos 的序列（作为输入）
        # decoder_targets  = 去掉最前面 sos 的序列（作为要预测的标准答案）
        # [:, :-1] ： 取batch_size行的所有元素，从第0列开始，到倒数第二个列（-1）结束。-1 表示最后一个数（不包含）
        # [:, 1:] ： 取batch_size行的所有元素，从第1列开始，到最后
        decoder_inputs = targets[:, :-1]  # decoder_inputs.shape: [batch_size, seq_len] seq_len=tgt_seq_len-1
        decoder_targets = targets[:, 1:]  # decoder_targets.shape: [batch_size, seq_len] seq_len=tgt_seq_len-1
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
        # src_pad_mask 标记中文侧的 <PAD> 位置，让编码器忽略填充。
        '''
            model.zh_embedding.padding_idx = zh_tokenizer.pad_token_index = 0 (是一个int的整数)
            如果 encoder_inputs = [[3, 45, 12, 0, 0],
                                    [7,  0,  0, 0, 0]]
            那么 encoder_inputs == model.zh_embedding.padding_idx
                                = [[False, False, False, True, True],
                                    [True, True, True, True, True]]
        '''
        src_pad_mask = (encoder_inputs == model.zh_embedding.padding_idx)
        # tgt_mask 是「下三角掩码」，训练时（teacher forcing）也要遵守自回归规则，防止偷看未来的目标词。
        '''
            generate_square_subsequent_mask 是 nn.Transformer 的方法
            用于 生成一个 下三角 + 上三角负无穷"的掩码矩阵
            generate_square_subsequent_mask(sz) 返回一个形状 [sz, sz] 的 float 张量
                - 主对角线及 以下 （ j <= i ，即每个位置能看到自己和之前的词）：值为 0
                - 主对角线 以上 （ j > i ，未来的词）：值为 -inf
            比如 sz = 4 时生成的就是：
                [[0, -inf, -inf, -inf],
                [0,    0, -inf, -inf],
                [0,    0,    0, -inf],
                [0,    0,    0,    0]]
        '''
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1])
        decoder_outputs = model(encoder_inputs, decoder_inputs, src_pad_mask, tgt_mask)
        # decoder_outputs.shape: [batch_size, seq_len, en_vocab_size]
        # 是 logits（未经过 softmax 的原始打分）

        # 展平成 [batch_size*seq_len, en_vocab_size] 与 [batch_size*seq_len]，再计算交叉熵损失。
        loss = loss_fn(decoder_outputs.reshape(-1, decoder_outputs.shape[-1]), decoder_targets.reshape(-1))

        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 清空梯度
        optimizer.zero_grad()
        # 记录损失
        total_loss += loss.item()
        # total_loss / len(dataloader) 是一个平均损失
    return total_loss / len(dataloader)


def train():
    # 1. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 数据
    dataloader = get_dataloader()
    # 3. 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    # 4. 模型
    model = TranslationModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_token_index,
                            en_tokenizer.pad_token_index).to(device)
    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 7. TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'Loss: {loss:.4f}')

        # 记录到Tensorboard
        writer.add_scalar('Loss', loss, epoch)

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
            print('保存模型')

    writer.close()


if __name__ == '__main__':
    train()
