import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator
from .text_model import TextFeatureExtractor

class LightweightChannelClassifier:
    def __init__(self):
        # 加载轻量级文本模型
        self.text_extractor = TextFeatureExtractor()
        
        # 加载预训练分类器
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'rf_classifier.pkl')
        self.classifier = joblib.load(model_path)
        
        # 加载分类映射
        with open(os.path.join(os.path.dirname(__file__), 'category_map.json'), 'r') as f:
            self.category_map = json.load(f)
        
        # 缓存已处理的频道名称
        self.cache = {}
    
    def predict(self, channel_name):
        """预测频道类别 - 轻量级版本"""
        # 检查缓存
        if channel_name in self.cache:
            return self.cache[channel_name]
        
        # 特殊频道优先处理
        if "春晚" in channel_name: 
            result = "cw"
        elif "直播中国" in channel_name: 
            result = "zb"
        elif any(kw in channel_name.lower() for kw in ["mtv", "music", "音樂", "演唱会"]): 
            result = "mv"
        elif any(kw in channel_name.lower() for kw in ["radio", "广播", "fm", "am"]): 
            result = "radio"
        elif any(kw in channel_name for kw in ["回看", "重播", "回放", "录像"]): 
            result = "lx"
        else:
            # 使用AI模型预测
            features = self.text_extractor.extract(channel_name)
            prediction = self.classifier.predict([features])[0]
            result = self.category_map[str(prediction)]
        
        # 缓存结果
        self.cache[channel_name] = result
        return result
    
    def predict_batch(self, channel_names):
        """批量预测频道类别"""
        results = []
        for name in channel_names:
            results.append(self.predict(name))
        return results

class TextFeatureExtractor:
    """轻量级文本特征提取器"""
    def __init__(self):
        # 使用轻量级sentence-transformers模型
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def extract(self, text):
        """提取文本特征向量 (384维)"""
        return self.model.encode([text])[0]
