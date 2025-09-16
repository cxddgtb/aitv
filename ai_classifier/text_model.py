import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class TextFeatureExtractor:
    """文本特征提取器"""
    def __init__(self, model_path=None):
        # 使用轻量级sentence-transformers模型
        if model_path and os.path.exists(model_path):
            self.model = SentenceTransformer(model_path)
        else:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def extract(self, text):
        """提取文本特征向量 (384维)"""
        return self.model.encode([text])[0]

    def extract_batch(self, texts):
        """批量提取文本特征向量"""
        return self.model.encode(texts)

def load_category_config():
    """加载分类配置"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'category_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_training_data():
    """准备训练数据"""
    config = load_category_config()
    data = []
    labels = []

    # 添加各类别样本
    for category_type in config.values():
        for cat_id, info in category_type.items():
            # 添加关键词
            for keyword in info.get("keywords", []):
                data.append(keyword)
                labels.append(cat_id)

            # 添加频道名称
            for name in info.get("dictionary", []):
                data.append(name)
                labels.append(cat_id)

    return data, labels

