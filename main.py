import os
import time
import json
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_classifier.classifier import LightweightChannelClassifier
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import queue

# --- 全局配置与初始化 ---
timestart = datetime.now()
print(f"🚀 AI频道分类系统启动 @ {timestart.strftime('%Y-%m-%d %H:%M:%S')}")

# 初始化AI分类器
classifier = LightweightChannelClassifier()

# ... (保留之前的配置和函数) ...

def main():
    # ... (保留之前的初始化步骤) ...
    
    print(f"\n➡️ 步骤 4/5: AI智能分类 {len(valid_channels)} 个频道...")
    
    # 分批处理避免内存溢出
    batch_size = 200
    categorized_lists = {cat_id: [] for cat_type in CATEGORY_CONFIG.values() for cat_id in cat_type}
    categorized_lists.update({'cw': [], 'zb': [], 'mv': [], 'radio': [], 'lx': [], 'other': []})
    
    # 提取频道名称列表
    channel_names = [name for name, _ in valid_channels]
    
    # 分批处理
    for i in range(0, len(channel_names), batch_size):
        batch_names = channel_names[i:i+batch_size]
        categories = classifier.predict_batch(batch_names)
        
        for j, category in enumerate(categories):
            name = batch_names[j]
            url = valid_channels[i+j][1]
            categorized_lists[category].append(f"{name},{url}")
        
        print(f"✅ 已分类: {min(i+batch_size, len(valid_channels))}/{len(valid_channels)} 个频道")
    
    # ... (保留后续步骤) ...
