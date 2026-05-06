from src.RelationGraph.func.model.lora.use import lora_calc_proba
from src.RelationGraph.func.utils.config import dataset_path

from datasets import DatasetDict, concatenate_datasets

# Step 1：取出数据
dataset = DatasetDict.load_from_disk(dataset_path)
combined_dataset = concatenate_datasets([
    dataset['train'],
    dataset['validation'],
    dataset['test']
])
texts = combined_dataset['text']
y = combined_dataset['label_id']

# Step 2：计算亲缘矩阵
lora_calc_proba(texts, y)
