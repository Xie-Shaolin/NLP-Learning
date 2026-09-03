import torch
from tqdm import tqdm


import config
from model import ReviewAnalyzeModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    """
    在测试集上评估模型，返回准确率。
    :param model: 已加载权重的模型。
    :param test_dataloader: 测试集 DataLoader。
    :param device: 运行设备（cuda 或 cpu）。
    :return: 准确率（正确预测数 / 总样本数），取值范围 [0,1]。
    """
    # 统计样本总数与预测正确的样本数
    total_count = 0
    correct_count = 0
    # 使用 tqdm 显示评估进度条
    for inputs in tqdm(test_dataloader, desc='评估'):
        # 从 batch 中取出并移除标签（pop 后 inputs 里只剩模型输入字段）
        labels = inputs.pop('labels').tolist()
        # 将模型输入张量移动到指定设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 调用批量预测得到概率，shape: [batch_size]，例如 [0.1, 0.2, 0.9, 0.3]
        batch_result = predict_batch(model, inputs)

        # 逐样本比较预测结果与真实标签，统计正确个数
        # zip() 是 Python 的 内置函数 ，作用是： 把多个可迭代对象"按位置"打包成元组，每个元组包含来自每个可迭代对象的对应元素。
        for result, target in zip(batch_result, labels):
            # 概率 > 0.5 判为正向（1），否则判为负向（0）
            result = 1 if result > 0.5 else 0
            if result == target:
                correct_count += 1
            total_count += 1
    # 返回整体准确率
    return correct_count / total_count


def run_evaluate():
    """评估入口：加载设备、模型、测试集，计算并打印准确率。"""
    # 准备资源
    # 1. 确定设备：有 GPU 则用 GPU，否则回退到 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载模型结构并载入训练好的权重
    model = ReviewAnalyzeModel().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    # 3. 加载测试集 DataLoader
    test_dataloader = get_dataloader(train=False)

    # 4. 执行评估逻辑，得到准确率
    acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"acc: {acc}")


if __name__ == '__main__':
    run_evaluate()
