# ========== 评估模块 ==========
# 用测试集评估模型质量，指标为 BLEU（衡量机器翻译与人工参考译文的相似度，越高越好）。
import torch
from nltk.translate.bleu_score import corpus_bleu

import config
from model import TranslationModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import ChineseTokenizer, EnglishTokenizer


def evaluate(model, test_dataloader, device, en_tokenizer):
    """在测试集上评估模型，计算整个语料库的 BLEU 分数。

    :param model: 训练好的模型
    :param test_dataloader: 测试数据加载器
    :param device: 计算设备
    :param en_tokenizer: 英文分词器（用于获取 <eos> 索引）
    :return: 语料级 BLEU 分数（float，范围约 0~1）
    """
    predictions = []   # 模型译文（token id 列表）
    # predictions: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]
    references = []    # 人工参考译文（token id 列表）
    # references: [[[*,*,*,*,*]],[[*,*,*,*]],[[*,*,*]]]

    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        # inputs.shape: [batch_size, seq_len]
        targets = targets.tolist()
        # targets.tolist()：把张量转成嵌套列表，方便做纯 Python 的截断操作
        # targets: [[sos,*,*,*,*,*,eos],[sos,*,*,*,*,eos,pad],[sos,*,*,*,eos,pad,pad]]

        batch_result = predict_batch(model, inputs, en_tokenizer)
        # batch_result: [[*,*,*,*,*],[*,*,*,*],[*,*,*]]

        predictions.extend(batch_result)
        # 构造参考译文：去掉开头的 <sos>，截到第一个 <eos> 为止（eos 本身也不保留）。
        # 外层套一层 [ ] 是因为 BLEU 允许一个译文对应多条参考，这里只有一条；
        # target.index(eos) 返回 eos 首次出现的下标；切片 target[1:pos] 保留 [1, pos) 区间。
        references.extend([
                [target[1:target.index(en_tokenizer.eos_token_index)]] for target in targets
            ])

    # corpus_bleu(list_of_references, hypotheses)：语料级 BLEU。
    #   list_of_references：[[ref1...], [ref2...]]（每个元素的参考本身是列表的列表）
    #   hypotheses：[[token...], ...]（模型译文）
    '''
        BLEU（Bilingual Evaluation Understudy）衡量的是 模型生成的译文与人工参考译文之间的相似度 ，
            取值范围 0~1 （越接近 1 越好）。
            核心思想是统计预测译文中有多少个 n-gram（连续 n 个词）能在参考译文中找到匹配。
        函数签名与参数结构
            corpus_bleu(list_of_references, 
                        hypotheses, weights=(0
                        .25, 0.25, 0.25, 0.25),
                        smoothing_function=None, ...)
            list_of_references: 每个 样本 对应一个"参考译文列表"（一个译文允许有多条参考）
                [
                    [[ref1a], [ref1b]], 
                    [[ref2a]], ...
                ]
                list_of_references 是 三层嵌套 （样本 → 参考列表 → 词序列），
            hypotheses: 每个样本对应 一条 模型预测译文
                [[tok...], [tok...], ...]
                hypotheses 是 两层嵌套 （样本 → 词序列）
    '''
    return corpus_bleu(references, predictions)


def run_evaluate():
    """评估主流程：加载词表/模型/测试集，输出 BLEU 分数。"""
    # 准备资源
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2.词表
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print("词表加载成功")

    # 3. 模型
    model = TranslationModel(zh_tokenizer.vocab_size, en_tokenizer.vocab_size, zh_tokenizer.pad_token_index,
                            en_tokenizer.pad_token_index).to(device)
    # torch.load 加载权重文件，load_state_dict 灌入模型（键值一一对应）
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    # 4. 数据集
    # train=False 表示加载测试集 test.jsonl
    test_dataloader = get_dataloader(train=False)

    # 5.评估逻辑
    bleu = evaluate(model, test_dataloader, device, en_tokenizer)
    print("评估结果")
    print(f"bleu: {bleu}")


if __name__ == '__main__':
    # 仅直接运行本文件时启动评估
    run_evaluate()
