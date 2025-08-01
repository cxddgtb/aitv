import os
import time
import json
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_classifier.classifier import ChannelClassifier
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 初始化AI分类器
classifier = ChannelClassifier()

# ... (之前的配置和函数保持不变) ...

def classify_with_ai(channel_name, channel_url):
    """使用AI模型分类频道"""
    # 从EPG获取台标URL
    logo_url = f"https://epg.112114.xyz/logo/{channel_name}.png"
    
    # 使用AI分类器
    category = classifier.predict(channel_name, logo_url)
    
    # 特殊类别后处理
    if "春晚" in channel_name: 
        return "cw"
    if "直播中国" in channel_name: 
        return "zb"
    if any(kw in channel_name.lower() for kw in ["mtv", "music", "音樂", "演唱会"]): 
        return "mv"
    if any(kw in channel_name.lower() for kw in ["radio", "广播", "fm", "am"]): 
        return "radio"
    
    return category

def main():
    # ... (之前的初始化步骤保持不变) ...
    
    print(f"\n➡️ 步骤 4/5: AI智能分类 {len(valid_channels)} 个频道...")
    
    # 分批处理避免内存溢出
    batch_size = 100
    categorized_lists = {cat_id: [] for cat_type in CATEGORY_CONFIG.values() for cat_id in cat_type}
    categorized_lists.update({'cw': [], 'zb': [], 'mv': [], 'radio': [], 'lx': [], 'other': []})
    
    for i in range(0, len(valid_channels), batch_size):
        batch = valid_channels[i:i+batch_size]
        categories = classifier.predict_batch([(name, f"https://epg.112114.xyz/logo/{name}.png") for name, _ in batch])
        
        for (name, url), category in zip(batch, categories):
            categorized_lists[category].append(f"{name},{url}")
        
        print(f"✅ 已分类: {min(i+batch_size, len(valid_channels))}/{len(valid_channels)} 个频道")
    
    # ... (后续步骤保持不变) ...
