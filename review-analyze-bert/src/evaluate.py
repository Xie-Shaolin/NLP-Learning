import torch
from tqdm import tqdm


import config
from model import ReviewAnalyzeModel
from dataset import get_dataloader
from predict import predict_batch
from src.tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    total_count = 0
    correct_count = 0
    for inputs in tqdm(test_dataloader, desc='评估'):
        labels = inputs.pop('labels').tolist()
        inputs =  {k: v.to(device) for k,v in inputs.items()}

        batch_result = predict_batch(model, inputs)
        # batch_result.shape: [batch_size] e.g. [0.1, 0.2, 0.9, 0.3]

        for result, target in zip(batch_result, labels):
            result = 1 if result > 0.5 else 0
            if result == target:
                correct_count += 1
            total_count += 1
    return correct_count / total_count


def run_evaluate():
    # 准备资源
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 模型
    model = ReviewAnalyzeModel().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    # 3. 数据集
    test_dataloader = get_dataloader(train=False)

    # 4.评估逻辑
    acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print(f"acc: {acc}")


if __name__ == '__main__':
    run_evaluate()
