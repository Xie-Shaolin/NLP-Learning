from pathlib import Path

# 项目根目录：本文件位于 src/ 下，向上两级即回到项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 原始数据目录（未处理的 CSV 文件存放于此）
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
# 处理后的数据集目录（切分并编码后的训练集/测试集存放于此）
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
# 日志目录（TensorBoard 事件文件存放于此）
LOGS_DIR = ROOT_DIR / "logs"
# 模型目录（训练好的模型权重 best.pt、词表等存放于此）
MODELS_DIR = ROOT_DIR / "models"
# 预训练模型目录（bert-base-chinese 预训练权重存放于此）
PRE_TRAINED_DIR = ROOT_DIR / "pretrained"

# ===== 训练相关超参数 =====
SEQ_LEN = 128          # 输入序列最大长度（分词后截断或填充到该长度）
BATCH_SIZE = 16        # 每个训练/评估批次的样本数量
EMBEDDING_DIM = 128    # 词向量维度（当前 BERT 方案中未直接使用，保留备用）
HIDDEN_SIZE = 256      # 隐藏层维度（当前 BERT 方案中未直接使用，保留备用）
LEARNING_RATE = 1e-5   # 优化器学习率
EPOCHS = 20            # 训练轮数