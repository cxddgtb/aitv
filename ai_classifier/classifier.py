import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.ensemble import RandomForestClassifier
from .text_model import TextFeatureExtractor
from .image_processor import ImageFeatureExtractor

class ChannelClassifier:
    def __init__(self):
        # 初始化模型
        self.text_extractor = TextFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        
        # 融合分类器
        self.fusion_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        
        # 类别映射
        self.category_map = {
            0: "ys", 1: "ws", 2: "ty", 3: "jp", 4: "kr", 
            5: "us", 6: "hk", 7: "tw", 8: "sh", 9: "bj",
            10: "gd", 11: "zj", 12: "yl", 13: "se", 14: "news",
            15: "kj", 16: "mv", 17: "dy", 18: "lx", 19: "other"
        }
        
        # 加载预训练模型
        self.load_model()
    
    def load_model(self):
        """加载预训练模型权重"""
        # 实际部署时从Hugging Face Hub加载
        pass
    
    def extract_features(self, channel_name, logo_url=None):
        """提取多模态特征"""
        # 文本特征
        text_features = self.text_extractor.extract(channel_name)
        
        # 图像特征
        image_features = np.zeros(512)
        if logo_url:
            try:
                image_features = self.image_extractor.extract(logo_url)
            except Exception:
                pass
        
        # 合并特征
        return np.concatenate([text_features, image_features])
    
    def predict(self, channel_name, logo_url=None):
        """预测频道类别"""
        features = self.extract_features(channel_name, logo_url)
        prediction = self.fusion_classifier.predict([features])[0]
        return self.category_map[prediction]
    
    def predict_batch(self, channels):
        """批量预测"""
        features = []
        for name, logo in channels:
            features.append(self.extract_features(name, logo))
        
        predictions = self.fusion_classifier.predict(features)
        return [self.category_map[p] for p in predictions]

class TextFeatureExtractor:
    """文本特征提取器"""
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        
    def extract(self, text):
        """提取文本特征向量"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

class ImageFeatureExtractor:
    """图像特征提取器"""
    def __init__(self):
        self.model = resnet18(pretrained=True)
        self.model.fc = torch.nn.Identity()  # 移除最后一层
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract(self, image_url):
        """从URL提取图像特征"""
        response = requests.get(image_url, stream=True, timeout=5)
        image = Image.open(response.raw)
        image = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(image)
        return features.squeeze().numpy()
