# ========== 项目全局配置 ==========
# 本文件集中管理所有超参数与文件路径，其他模块统一从这里读取，避免魔法数字散落各处。

# from pathlib import Path
# Path 是 Python 标准库中面向对象的路径操作类。
# 与字符串拼接路径不同，Path 支持 "/" 运算符拼接，且在不同操作系统（Windows/Linux/macOS）上自动使用正确的分隔符。
from pathlib import Path

# ROOT_DIR：项目根目录。
# __file__ 是当前文件（config.py）的绝对路径；.parent 取其父目录（src），再 .parent 取上一级（项目根目录）。
# 这样无论从哪里运行脚本，都能准确定位到项目根目录。
ROOT_DIR = Path(__file__).parent.parent

# 数据目录：原始数据放在 data/raw 下
# "/" 运算符在这里等价于 os.path.join，用于拼接路径
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
# 处理后的数据（train.jsonl / test.jsonl）放在 data/processed 下
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
# TensorBoard 日志目录
LOGS_DIR = ROOT_DIR / "logs"
# 模型权重保存目录
MODELS_DIR = ROOT_DIR / "models"

# ========== 模型超参数 ==========
# 最大序列长度：句子超过该长度会被截断（预测阶段自回归最多生成这么多个 token）
MAX_SEQ_LENGTH = 128
# 批大小：一次送入 GPU 的样本数量，越大越占显存但训练更快
BATCH_SIZE = 64
# 词嵌入维度：每个 token 映射成的稠密向量长度
EMBEDDING_DIM = 128
# GRU 隐藏层维度：隐藏状态向量的长度
HIDDEN_SIZE = 256
# 学习率：控制每步参数更新的步长（1e-3 即 0.001，科学计数法写法）
LEARNING_RATE = 1e-3
# 训练轮数：整个数据集被遍历的次数
EPOCHS = 50
