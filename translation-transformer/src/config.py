from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# 路径
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

# 训练参数
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50

# ========== 模型结构（transformer 版相对 attention 版的核心差异） ==========
# attention 版基于 GRU，需要 EMBEDDING_DIM 和 HIDDEN_SIZE 两个独立维度；
# transformer 版把嵌入维度与模型内部维度统一为 DIM_MODEL，多头注意力、前馈层都围绕它展开。
DIM_MODEL = 128  # 每个 token 的向量维度（相当于 attention 版的 EMBEDDING_DIM）
NUM_HEADS = 4  # 多头注意力的头数，需能被 DIM_MODEL 整除（128 / 4 = 32）
NUM_ENCODER_LAYERS = 2  # 编码器堆叠的 Transformer 层数
NUM_DECODER_LAYERS = 2  # 解码器堆叠的 Transformer 层数
