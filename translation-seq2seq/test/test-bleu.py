# ========== BLEU 计算示例脚本 ==========
# 用两对"预测译文 vs 参考译文"验证 nltk 的 corpus_bleu 用法，
# 便于理解 evaluate.py 中 BLEU 分数的数据结构要求。
from nltk.translate.bleu_score import corpus_bleu

# pre1：模型预测的第 1 条译文（token 列表）
pre1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
        'ensures', 'that', 'the', 'military', 'always',
        'obeys', 'the', 'commands', 'of', 'the', 'party']

# ref1a：第 1 条译文对应的人工参考译文。
# 注意：参考比预测多了几个 'aaa'——BLEU 统计 n-gram 时参考的长度更长，
# 对过短的译文有"长度惩罚"，多余词不会提升分数（默认 ngram_order=4、平滑策略等）。
ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
         'ensures', 'that', 'the', 'military', 'always',
         'obeys', 'the', 'commands', 'of', 'the', 'party', 'aaa', 'aaa', 'aaa']

# pre2 / ref2a：第 2 对预测与参考
pre2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
        'interested', 'in', 'world', 'history']
ref2a = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
         'interested', 'in', 'world', 'history', 'aaa']

# corpus_bleu 的两个参数：
#   list_of_references：每个样本的参考译文列表，形如 [[ref1a], [ref2a]]
#     （一个译文可对应多条参考，所以每个元素是"参考的列表"）
#   hypotheses：预测译文列表，形如 [pre1, pre2]
list_of_references = [[ref1a], [ref2a]]
predictions = [pre1, pre2]
print(corpus_bleu(list_of_references, predictions))
