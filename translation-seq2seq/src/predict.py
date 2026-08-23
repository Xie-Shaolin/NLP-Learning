# ========== 预测模块 ==========
# 提供两类功能：
#   1. predict_batch：对一批中文句子做自回归生成（推理模式）；
#   2. predict / run_predict：单条中文句子的命令行交互式翻译。
import torch

import config
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(model, inputs, en_tokenizer):
    """批量预测：对一批中文句子自回归地逐个生成英文单词，直到遇到 <eos> 或达到最大长度。

    与训练不同，推理时没有标准答案，必须"用模型自己上一步的输出作为下一步的输入"（自回归）。

    :param model: 训练好的 TranslationModel
    :param inputs: 中文索引序列，shape: [batch_size, seq_len]
    :param en_tokenizer: 英文分词器（用于取 <sos>/<eos> 索引）
    :return: 预测结果列表，形如 [[token1, token2, ...], [...], ...]（每个样本长度可能不同）
    """
    model.eval()  # 切换到评估模式（关闭训练专用行为，如 Dropout）
    with torch.no_grad():
        # torch.no_grad() 上下文管理器：关闭梯度计算。
        # 推理不需要反向传播，关闭后省显存、加速，且不会污染参数的 .grad。

        # 编码：中文句子 -> 语义向量
        context_vector = model.encoder(inputs)
        # context_vector.shape: [batch_size, hidden_size]

        # 解码
        batch_size = inputs.shape[0]
        device = inputs.device

        # 隐藏状态：unsqueeze(0) 增加层维度，匹配 GRU 的 (层数, batch, hidden) 格式
        decoder_hidden = context_vector.unsqueeze(0)
        # decoder_hidden.shape: [1, batch_size, hidden_size]

        # 初始输入：每个样本都用 <sos> 开头。
        # torch.full([batch_size, 1], value)：创建形状为 [batch_size, 1]、
        # 所有元素都是 value 的张量（value 是标量，此处是 sos 索引）。
        decoder_input = torch.full([batch_size, 1], en_tokenizer.sos_token_index, device=device)
        # decoder_input.shape: [batch_size, 1]

        # 预测结果缓存：收集每一步生成的所有 token
        generated = []

        # 记录每个样本是否已生成 <eos>（生成完毕）。
        # torch.full 填充 False；|= 是按位或赋值，一旦置 True 就永久为 True。
        is_finished = torch.full([batch_size], False, device=device)

        # 自回归生成：最多生成 MAX_SEQ_LENGTH 步，防止模型不收敛导致死循环
        for i in range(config.MAX_SEQ_LENGTH):
            # 单步解码：输入上一步的词，得到本步预测分布和更新后的隐状态
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            # decoder_output.shape: [batch_size, 1, vocab_size]

            # 保存预测结果
            # torch.argmax(张量, dim=-1)：沿最后一个维度（词表维）取最大值的下标，
            # 即得分最高的候选词索引 = 贪心解码（不做 beam search）。
            next_token_indexes = torch.argmax(decoder_output, dim=-1)
            # next_token_indexes.shape: [batch_size, 1]
            generated.append(next_token_indexes)

            # 更新输入：把本步预测的词作为下一步的输入（自回归的核心）
            decoder_input = next_token_indexes

            '''
            |= 是 按位或赋值运算符 ，等价于 a = a | b
                在这里， a 和 b 都是 布尔张量 ，所以对布尔值来说，按位或就是 逻辑或 。
                它的核心语义是： 一旦某个位置被置为 True ，就永远保持 True ，不会被后续的 False 覆盖
                即"一旦结束就永远结束"。
            is_finished = is_finished | (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
                        = [False, True, False] | [True,  False,  False] = [True, True, False]
            '''
            # 判断是否应该结束
            # squeeze(1) 去掉第 1 维（尺寸为 1），[batch_size, 1] -> [batch_size]；
            # 与 eos 索引逐元素比较，得到布尔张量；
            # |= 把"本步已结束"的样本标记进 is_finished。
            is_finished |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            # .all()：只有当整个 batch 都生成了 eos 才提前退出循环
            '''
                is_finished.all() 是 PyTorch Tensor 的 归约（reduction）方法 ，属于"逻辑归约"这一族。
                它把整个张量里所有元素做 逻辑与（AND） ，返回一个 标量张量 。
                    .all()：逻辑与（AND）, 只有当所有元素都为 True 才返回 True，否则返回 False。
                    .any()：逻辑或（OR）, 只要有一个元素为 True 就返回 True，否则返回 False。
            '''
            if is_finished.all():
                break

        # 处理预测结果
        # 把各步收集的张量沿时间步维拼接
        # torch.cat(generated, dim=1)
        generated_tensor = torch.cat(generated, dim=1)
        # generated_tensor.shape: [batch_size, seq_len]

        # .tolist()：张量转成嵌套 Python 列表，方便后续处理
        generated_list = generated_tensor.tolist()
        # generated_list：[[*,*,*,*,*],[*,*,*,eos,*],[*,*,eos,*,*]]

        # 去掉eos之后的token id
        # 每个样本在 eos 处截断（eos 本身也不保留）
        for index, sentence in enumerate(generated_list):
            # 如果 en_tokenizer.eos_token_index 在 sentence 中
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)  # 找到 eos 首次出现的位置
                generated_list[index] = sentence[:eos_pos]              # 切片截断，只留 eos 之前的词
        # generated_list：[[*,*,*,*,*],[*,*,*],[*,*]]
        #  # generated_list = [I, love, you]   # 没有 sos，也没有 eos
        return generated_list


def predict(text, model, zh_tokenizer, en_tokenizer, device):
    """单条中文句子的翻译：文本 -> 预测索引序列 -> 英文句子。

    :param text: 中文句子字符串
    :param model: 训练好的模型
    :param zh_tokenizer: 中文分词器（文本编码用）
    :param en_tokenizer: 英文分词器（索引解码用）
    :param device: 计算设备
    :return: 翻译出的英文句子字符串
    """
    # 1. 处理输入
    # encode(text) 默认不加 <sos>/<eos>（编码器输入不需要起止符）
    indexes = zh_tokenizer.encode(text)
    # torch.tensor([indexes])：再套一层方括号，把一维列表变成形状 [1, seq_len] 的二维张量
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)  # 迁移到 GPU/CPU
    # input_tensor.shape: [batch_size, seq_len]（此处 batch_size=1）

    # 2.预测逻辑
    batch_result = predict_batch(model, input_tensor, en_tokenizer)
    # 只有一个样本，取 [0]；decode 把索引列表还原成英文句子
    return en_tokenizer.decode(batch_result[0])


def run_predict():
    """命令行交互式翻译：加载资源后进入循环，读取用户输入并打印译文。"""
    # 准备资源
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print("分词器加载成功")

    # 3. 模型
    model = TranslationModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_token_index,
                            en_tokenizer.pad_token_index).to(device)
    # load_state_dict：把训练保存的 best.pt 权重加载进模型（键一一对应）
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    print("欢迎使用中英翻译模型(输入q或者quit退出)")

    # 交互循环：input() 阻塞等待用户在终端输入一行文字
    while True:
        user_input = input("中文：")
        if user_input in ['q', 'quit']:      # in 判断是否命中退出关键字
            print("欢迎下次再来")
            break                            # break 跳出 while 循环
        if user_input.strip() == '':         # strip() 去掉首尾空格后判断是否为空
            print("请输入内容")
            continue                         # continue 跳过本次，继续下一次输入

        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print("英文：", result)


if __name__ == '__main__':
    # 仅直接运行本文件时启动交互式翻译
    run_predict()
