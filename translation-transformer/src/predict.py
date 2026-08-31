import torch

import config
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(model, inputs, en_tokenizer):
    """
    批量预测
    :param model: 模型
    :param inputs: 输入,shape:[batch_size, seq_len]
    :return: 预测结果: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]
    """
    model.eval()
    with torch.no_grad():
        # 编码
        # src_pad_mask 标记输入中文里哪些位置是 <PAD>，让编码器忽略这些填充位置。
        # 【与 attention 版的差异】attention 版编码器返回 (encoder_outputs, context_vector) 两个值；
        # transformer 版只需一次 encode 得到 memory（编码器的完整上下文表示），不再需要 context_vector。
        src_pad_mask = (inputs == model.zh_embedding.padding_idx)
        memory = model.encode(inputs, src_pad_mask)
        # memory.shape: [batch_size, src_seq_len, d_model]

        # 解码
        batch_size = inputs.shape[0]
        device = inputs.device

        # 解码起始：decoder_input 初始只含一个 <SOS>，之后每轮生成一个词并拼接到末尾。
        '''
            这里非常关键，
                训练时（train.py）：整句一次性喂入，并行计算，model.decode 只调用 一次 ，Transformer 内部并行算出所有位置的输出
                    decoder_inputs = targets[:, :-1] 
                    decoder_output = model.decode(decoder_input, memory, tgt_mask, src_pad_mask)
                推理时（predict.py）：逐词生成，循环多次
                    decoder_input = torch.full([batch_size, 1]... : 初始只有 <sos>
                    for i in range(config.MAX_SEQ_LENGTH):     # ← 循环！最多 128 次
                        decoder_output = model.decode(decoder_input, memory, tgt_mask, src_pad_mask)  # ← 每轮都调用一次
                        next_token_indexes = torch.argmax(decoder_output[:, -1, :], dim=-1, keepdim=True)  # 只取最后一步
                        decoder_input = torch.cat([decoder_input, next_token_indexes], dim=-1)   # ← 拼接新词，序列变长
        '''
        decoder_input = torch.full([batch_size, 1], en_tokenizer.sos_token_index, device=device)
        # decoder_input.shape: [batch_size, tgt_seq_len]（tgt_seq_len 会随着生成逐步变长）

        # 预测结果缓存
        generated = []

        # 记录每个样本是否已经生成结束符
        is_finished = torch.full([batch_size], False, device=device)

        # 自回归生成
        for i in range(config.MAX_SEQ_LENGTH):
            # 解码
            # tgt_mask 是「下三角掩码」，保证每个位置只能关注它自身及之前的 token（自回归，不能看未来）。
            # 【与 attention 版的差异】attention 版每次只把「当前一个词」喂给单步 GRU；
            # transformer 版把「已经生成的所有词」整体喂给解码器，再只取最后一个位置的输出作为本步结果。
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1])
            '''
                已经生成 eos 的句子，在循环里仍然会继续生成内容 。
                通常是重复的 eos 或无意义词，比如 [sos, I, love, you, eos, good, few]
                多生成的数字是"垃圾 token"，最后需要被 切掉 。
            '''
            decoder_output = model.decode(decoder_input, memory, tgt_mask, src_pad_mask)
            # decoder_output.shape: [batch_size, tgt_seq_len, en_vocab_size]

            # 保存预测结果：只取最后一步（最新生成的那个位置）的输出做 argmax，得到下一个 token。
            '''
                decoder_output[:, -1, :] —— 取最后一个位置的 logits
                    decoder_output 的形状是 [batch_size, tgt_seq_len, en_vocab_size] ，其中：
                        - : 第一维 = 取所有样本（batch）
                        - -1 第二维 = 取最后一个位置 （最新生成的那个词）
                        - : 第三维 = 取词表维度（所有词的分数）
                    假设 decoder_output 是 [2, 4, 5000] （2 个样本、当前已生成 4 个词、词表 5000），那么：decoder_output[:, -1, :]  →  shape 变为 [2, 5000]
                torch.argmax(..., dim=-1) —— 找出分数最高的词 
                    - dim=-1 表示沿**最后一维（词表维度）**取最大值所在的下标：
                    - keepdim=True —— 保持维度为 [batch_size, 1]
                
                举例分析：
                    decoder_output [2, 3, 5]   （当前已生成 3 个词，词表 5 个词）
                    样本0 logits:  [[0.1, 0.2, 0.3, 0.1, 0.1],      ← 位置0
                                    [0.5, 0.1, 0.2, 0.1, 0.1],      ← 位置1
                                    [0.1, 0.1, 0.7, 0.0, 0.1]]      ← 位置2（最后一步）
                    样本1 logits:  [[0.2, 0.1, 0.1, 0.5, 0.1],
                                    [0.1, 0.2, 0.1, 0.3, 0.3],
                                    [0.0, 0.1, 0.2, 0.1, 0.6]]      ← 位置2（最后一步）
                    decoder_output[:, -1, :]  →  只取每行最后一个位置
                        [[0.1, 0.1, 0.7, 0.0, 0.1],      ← 样本0 第2位置
                        [0.0, 0.1, 0.2, 0.1, 0.6]]      ← 样本1 第2位置
                    torch.argmax(dim=-1)  →  每个样本取分数最高的词下标
                        [2, 4]                            （样本0 选词2=0.7；样本1 选词4=0.6）
                    keepdim=True  →  [[2], [4]]           ← 形状 [2, 1]，供拼接
            '''
            next_token_indexes = torch.argmax(decoder_output[:, -1, :], dim=-1, keepdim=True)
            # next_token_indexes.shape: [batch_size, 1]
            generated.append(next_token_indexes)

            # 更新输入：把新生成的 token 拼到 decoder_input 末尾，供下一轮继续生成。
            decoder_input = torch.cat([decoder_input, next_token_indexes], dim=-1)

            # 判断是否应该结束
            '''
            |= 是 按位或赋值运算符 ，等价于 a = a | b
                在这里， a 和 b 都是 布尔张量 ，所以对布尔值来说，按位或就是 逻辑或 。
                它的核心语义是： 一旦某个位置被置为 True ，就永远保持 True ，不会被后续的 False 覆盖
                即"一旦结束就永远结束"。
            squeeze(1) 去掉第 1 维（尺寸为 1），[batch_size, 1] -> [batch_size]；
            is_finished = is_finished | (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
                        = [False, True, False] | [True,  False,  False] = [True, True, False]
            
            |= 就是 按位或赋值（bitwise OR assignment） ，
            等价于 is_finished = is_finished | ... ，具有"累积/记忆"效果。
                is_finished:  [False, False]  初始
                    第4轮 |=       [False, False] |[True,  False] = [True,  False] 
                    第5轮 |=      [True,  False] | [False,  True] = [True,  True] 
                    第5轮后检查 all()  → True，break
            '''
            is_finished |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            '''
                is_finished.all() 是 PyTorch Tensor 的 归约（reduction）方法 ，属于"逻辑归约"这一族。
                它把整个张量里所有元素做 逻辑与（AND） ，返回一个 标量张量 。
                    .all()：逻辑与（AND）, 只有当所有元素都为 True 才返回 True，否则返回 False。
                    .any()：逻辑或（OR）, 只要有一个元素为 True 就返回 True，否则返回 False。
            '''
            if is_finished.all():
                break

        # 处理预测结果
        # 整理预测结果形状
        # generated：[tensor([batch_size, 1])]
        # generated 是关于 Tensor 的 list
        generated_tensor = torch.cat(generated, dim=1)
        # generated_tensor.shape: [batch_size,seq_len]
        generated_list = generated_tensor.tolist()
        # generated_list：[[*,*,*,*,*],[*,*,*,eos,*],[*,*,eos,*,*]]

        # 去掉eos之后的token id
        for index, sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]
        # generated_list：[[*,*,*,*,*],[*,*,*],[*,*]]
        return generated_list


def predict(text, model, zh_tokenizer, en_tokenizer, device):
    # 1. 处理输入
    indexes = zh_tokenizer.encode(text)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [batch_size, seq_len]

    # 2.预测逻辑
    batch_result = predict_batch(model, input_tensor, en_tokenizer)
    return en_tokenizer.decode(batch_result[0])


def run_predict():
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
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    print("欢迎使用中英翻译模型(输入q或者quit退出)")

    while True:
        user_input = input("中文：")
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue

        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print("英文：", result)


if __name__ == '__main__':
    run_predict()
