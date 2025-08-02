import os
import json
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class FundPredictor:
    def __init__(self, model_path=None):
        """初始化基金预测器"""
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            random_state=42
        )
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        self.scaler = None
        
    def fetch_fund_data(self, fund_code, start_date=None, end_date=None):
        """获取基金历史数据"""
        # 如果没有指定日期，默认获取最近一年的数据
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # 使用天天基金网API获取数据
        url = f"http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code={fund_code}&sdate={start_date}&edate={end_date}&per=20"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # 解析HTML响应
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取表格数据
            table = soup.find('table', class_='w782')
            if not table:
                raise ValueError(f"无法找到基金{fund_code}的数据")
            
            # 解析表格行
            rows = table.find_all('tr')[1:]
            data = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 7:
                    date = cols[0].text.strip()
                    unit_value = float(cols[1].text.strip())
                    cumulative_value = float(cols[2].text.strip())
                    change_rate = float(cols[3].text.strip().replace('%', ''))
                    
                    data.append({
                        'date': date,
                        'unit_value': unit_value,
                        'cumulative_value': cumulative_value,
                        'change_rate': change_rate
                    })
            
            # 转换为DataFrame并按日期排序
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            return df
        except Exception as e:
            print(f"获取基金数据失败: {e}")
            return None

    def create_features(self, df, window_size=5):
        """创建特征工程"""
        # 复制数据以避免修改原始数据
        features = df.copy()
        
        # 添加移动平均线
        features[f'ma{window_size}'] = features['unit_value'].rolling(window=window_size).mean()
        
        # 添加动量指标
        features['momentum'] = features['unit_value'] - features['unit_value'].shift(window_size)
        
        # 添加波动率
        features['volatility'] = features['change_rate'].rolling(window=window_size).std()
        
        # 添加相对强弱指数(RSI)计算
        delta = features['unit_value'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window_size).mean()
        avg_loss = loss.rolling(window=window_size).mean()
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # 移除NaN值
        features = features.dropna()
        
        return features

    def prepare_training_data(self, features, target_col='unit_value', forecast_horizon=1):
        """准备训练数据"""
        # 创建目标变量（预测未来n天的净值）
        features['target'] = features[target_col].shift(-forecast_horizon)
        
        # 移除NaN值
        features = features.dropna()
        
        # 选择特征列
        feature_cols = ['unit_value', 'cumulative_value', 'change_rate', 'ma5', 'momentum', 'volatility', 'rsi']
        
        X = features[feature_cols]
        y = features['target']
        
        return X, y

    def train(self, X, y):
        """训练模型"""
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 预测测试集
        y_pred = self.model.predict(X_test)
        
        # 评估模型
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"模型训练完成！")
        print(f"测试集MAE: {mae:.4f}")
        print(f"测试集RMSE: {rmse:.4f}")
        
        return mae, rmse

    def predict(self, X):
        """预测基金净值"""
        return self.model.predict(X)

    def save_model(self, model_path='models/fund_predictor.pkl'):
        """保存模型"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"模型已保存到 {model_path}")

    def load_model(self, model_path='models/fund_predictor.pkl'):
        """加载模型"""
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"模型已从 {model_path} 加载")
        else:
            print(f"模型文件 {model_path} 不存在")

    def generate_trading_signal(self, current_value, predicted_value):
        """生成买卖信号"""
        # 简单的信号生成策略：如果预测净值高于当前净值1%以上，则买入；如果低于当前净值1%以上，则卖出
        threshold = 0.01
        
        if predicted_value > current_value * (1 + threshold):
            return '买入'
        elif predicted_value < current_value * (1 - threshold):
            return '卖出'
        else:
            return '持有'


def main():
    # 创建基金预测器实例
    predictor = FundPredictor()
    
    # 获取基金数据（以易方达消费行业股票基金为例）
    fund_code = '110022'
    df = predictor.fetch_fund_data(fund_code)
    
    if df is not None and not df.empty:
        # 创建特征
        features = predictor.create_features(df)
        
        # 准备训练数据
        X, y = predictor.prepare_training_data(features)
        
        # 训练模型
        predictor.train(X, y)
        
        # 保存模型
        predictor.save_model()
        
        # 预测最新数据
        latest_data = features.iloc[-1:][['unit_value', 'cumulative_value', 'change_rate', 'ma5', 'momentum', 'volatility', 'rsi']]
        predicted_value = predictor.predict(latest_data)[0]
        current_value = df.iloc[-1]['unit_value']
        
        # 生成买卖信号
        signal = predictor.generate_trading_signal(current_value, predicted_value)
        
        print(f"\n基金代码: {fund_code}")
        print(f"当前净值: {current_value:.4f}")
        print(f"预测净值: {predicted_value:.4f}")
        print(f"买卖信号: {signal}")


if __name__ == "__main__":
    main()