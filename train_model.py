import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from .text_model import TextFeatureExtractor
from .classifier import CATEGORY_CONFIG

def prepare_training_data():
    """准备训练数据（示例，实际需要真实数据）"""
    # 这里使用模拟数据，实际应用应使用标注好的数据集
    data = []
    labels = []
    
    # 添加各类别样本
    for category_type in CATEGORY_CONFIG.values():
        for cat_id, info in category_type.items():
            for name in info['dictionary']:
                data.append(name)
                labels.append(cat_id)
    
    # 添加一些常见频道
    common_channels = [
        "CCTV1 综合", "CCTV2 财经", "CCTV5 体育", 
        "湖南卫视", "浙江卫视", "东方卫视",
        "凤凰卫视", "Discovery探索", "国家地理"
    ]
    common_labels = ["ys", "ws", "ys", "ws", "ws", "ws", "hk", "kj", "kj"]
    
    data.extend(common_channels)
    labels.extend(common_labels)
    
    return data, labels

def train_classifier():
    """训练并保存分类器模型"""
    # 准备数据
    data, labels = prepare_training_data()
    
    # 创建标签映射
    unique_labels = list(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    # 保存标签映射
    with open('ai_classifier/category_map.json', 'w') as f:
        json.dump(id_to_label, f)
    
    # 转换标签
    y = np.array([label_to_id[label] for label in labels])
    
    # 提取特征
    extractor = TextFeatureExtractor()
    X = np.array([extractor.extract(text) for text in data])
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15,
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心
    )
    model.fit(X_train, y_train)
    
    # 评估模型
    accuracy = model.score(X_test, y_test)
    print(f"模型准确率: {accuracy:.2f}")
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/rf_classifier.pkl')
    print("模型保存成功")

if __name__ == "__main__":
    train_classifier()
