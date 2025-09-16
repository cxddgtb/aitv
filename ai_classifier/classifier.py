import os
import sys
import json
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator

# 添加当前目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_classifier.text_model import TextFeatureExtractor

class LightweightChannelClassifier:
    def __init__(self):
        # 加载轻量级文本模型
        self.text_extractor = TextFeatureExtractor()
        
        # 加载预训练分类器
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'rf_classifier.pkl')
        try:
            self.classifier = joblib.load(model_path)
        except Exception as e:
            print(f"警告: 无法加载模型文件 {model_path}: {e}")
            # 创建一个简单的分类器作为后备
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            self.classifier = RandomForestClassifier(n_estimators=10, random_state=42)
            # 使用虚拟数据拟合
            X = np.random.rand(10, 10)
            y = np.random.randint(0, 5, 10)
            self.classifier.fit(X, y)
        
        # 加载分类映射
        category_map_path = os.path.join(os.path.dirname(__file__), 'category_map.json')
        try:
            with open(category_map_path, 'r', encoding='utf-8') as f:
                self.category_map = json.load(f)
        except Exception as e:
            print(f"警告: 无法加载分类映射文件 {category_map_path}: {e}")
            # 使用默认分类映射
            self.category_map = {
                "0": "cctv", "1": "local", "2": "gat", "3": "asia", "4": "west",
                "5": "sports", "6": "news", "7": "kids", "8": "documentary",
                "9": "ent", "10": "movie", "11": "music", "12": "lx",
                "13": "cw", "14": "zb", "15": "radio", "16": "other"
            }
        
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

# TextFeatureExtractor 类已在 text_model.py 中定义
    """轻量级文本特征提取器"""
    def __init__(self):
        # 使用轻量级sentence-transformers模型
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def extract(self, text):
        """提取文本特征向量 (384维)"""
        return self.model.encode([text])[0]
