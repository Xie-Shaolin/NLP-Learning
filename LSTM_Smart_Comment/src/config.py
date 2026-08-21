"""
项目配置文件
集中管理项目中用到的所有路径和超参数，方便统一修改。
"""

from pathlib import Path

# 项目根目录：当前文件(src/config.py)的上上级目录，即 LSTM_Smart_Comment 目录
ROOT_DIR = Path(__file__).parent.parent

# 原始数据目录：存放未处理的原始 CSV 数据
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
# 预处理后数据目录：存放划分好、编码后的训练/测试数据(jsonl 格式)
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
# 日志目录：存放 TensorBoard 训练日志
LOGS_DIR = ROOT_DIR / "logs"
# 模型目录：存放词表(vocab.txt)和训练好的模型权重(best.pt)
MODELS_DIR = ROOT_DIR / "models"

# ============ 超参数 ============
# 每条评论编码后的固定长度：超出的截断，不足的用 <pad> 填充
SEQ_LEN = 128
# 每次送入模型训练的样本数量（批大小）
BATCH_SIZE = 64
# 词嵌入向量的维度：把每个词映射成一个 128 维的向量
EMBEDDING_DIM = 128
# LSTM 隐藏层神经元数量
HIDDEN_SIZE = 256
# 优化器 Adam 的学习率
LEARNING_RATE = 1e-3
# 训练的总轮数（整个训练集被完整遍历的次数）
EPOCHS = 20
