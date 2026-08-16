"""
全局配置模块
集中管理项目的目录路径和超参数配置
"""

from pathlib import Path


# 项目根目录：当前文件(config.py)的上两级目录
# __file__ 表示当前配置文件的路径
# .parent.parent 表示向上两级，即项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 原始数据目录：存放未经处理的原始数据文件
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

# 处理后数据目录：存放经过预处理、划分好的训练集和测试集
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# 日志目录：存放 TensorBoard 训练日志文件
LOGS_DIR = ROOT_DIR / "logs"

# 模型保存目录：存放训练好的模型权重文件和词表文件
MODELS_DIR = ROOT_DIR / "models"

# 序列长度：模型输入的历史上下文词的数量
# 即使用前 SEQ_LEN 个词来预测下一个词
SEQ_LEN = 5

# 批次大小：每次训练时并行处理的样本数量
# 较大的 batch_size 可以提高训练稳定性和并行效率，但需要更多内存
BATCH_SIZE = 64

# 词嵌入维度：每个词被映射为的向量空间的维度
# 维度越高，词表示能力越强，但参数量也越大
EMBEDDING_DIM = 128

# RNN隐藏层大小：循环神经网络隐藏状态的维度
# 控制模型记忆容量，越大越能捕捉复杂依赖，但训练难度也增加
HIDDEN_SIZE = 256

# 学习率：优化器更新参数时的步长大小
# 1e-3 (0.001) 是 Adam 优化器常用的默认学习率
LEARNING_RATE = 1e-3

# 训练轮数：完整遍历训练集的次数
# 每个 epoch 会使用全部训练数据对模型进行一次更新
EPOCHS = 10
