from datasets import Dataset
from datasets import ClassLabel
dataset = Dataset.from_dict({
    "text": [
        "I love this movie.",
        "This movie is terrible."
    ],
    "label": [1, 0]
})

print(dataset)
print(dataset.features)
print(dataset[0])
print(dataset[1])
print(dataset.features["label"])
'''
Dataset({
    features: ['text', 'label'],
    num_rows: 2
})
{'text': Value('string'), 'label': Value('int64')}
{'text': 'I love this movie.', 'label': 1}
{'text': 'This movie is terrible.', 'label': 0}
Value('int64')
'''
# label_feature = dataset.features["label"]
# print(label_feature.int2str(0)) # 没有这个方法
# print(label_feature.int2str(1)) # 没有这个方法
# print(label_feature.str2int("negative")) # 没有这个方法
# print(label_feature.str2int("positive")) # 没有这个方法

print("-------------------------------------------------")

dataset = dataset.cast_column(
    "label",
    ClassLabel(names=["negative", "positive"])
)

print(dataset.features)
print(dataset[0])
print(dataset[1])
print(dataset.features["label"])
label_feature = dataset.features["label"]
print(label_feature.int2str(0))
print(label_feature.int2str(1))
print(label_feature.str2int("negative"))
print(label_feature.str2int("positive"))
'''
{'text': Value('string'), 'label': ClassLabel(names=['negative', 'positive'])}
{'text': 'I love this movie.', 'label': 1}
{'text': 'This movie is terrible.', 'label': 0}
ClassLabel(names=['negative', 'positive'])
negative
positive
0
1
'''