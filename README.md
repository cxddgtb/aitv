# AI学习型IPTV系统

这是一个智能IPTV频道分类系统，使用人工智能技术自动对电视频道进行分类和整理。

## 功能特点

- 自动获取和解析IPTV源
- 智能分类电视频道
- 自动检测频道可用性
- AI学习功能，可不断优化分类结果
- 支持多种输出格式（TXT和M3U）

## 安装说明

### 环境要求

- Python 3.8 或更高版本
- 推荐使用虚拟环境

### 安装步骤

1. 克隆或下载本项目
   ```bash
   git clone https://github.com/yourusername/aitv.git
   cd aitv
   ```

2. 创建并激活虚拟环境（推荐）
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

4. 训练AI模型（首次运行或需要更新模型时）
   ```bash
   python train_model.py
   ```

## 使用说明

### 基本使用

1. 编辑 `assets/urls.txt` 文件，添加IPTV源URL（每行一个URL）
2. 编辑 `assets/corrections_name.txt` 文件，添加频道名称纠错规则
3. 运行主程序
   ```bash
   python main.py
   ```

### 高级配置

#### 分类配置

编辑 `category_config.json` 文件可以自定义分类规则：

- `region_categories`: 地区分类（如央视频道、地方卫视等）
- `content_categories`: 内容分类（如体育赛事、新闻资讯等）
- `special_categories`: 特殊分类（如时移回看、春晚特辑等）

每个分类可以设置：
- `name`: 分类显示名称
- `keywords`: 关键词列表（用于匹配频道名称）
- `dictionary`: 频道名称字典（精确匹配）

#### AI模型训练

如果需要重新训练AI模型：

1. 准备训练数据（编辑 `manual_training.csv` 文件）
2. 运行训练脚本
   ```bash
   python train_model.py
   ```

### Docker部署

1. 构建镜像
   ```bash
   docker build -t aitv .
   ```

2. 运行容器
   ```bash
   docker run -v $(pwd)/output:/app/output aitv
   ```

## 输出文件

程序运行后会生成以下文件：

- `live.txt`: 文本格式的频道列表
- `live.m3u`: M3U格式的播放列表
- `live_lite.txt`: 精简版频道列表
- `live_lite.m3u`: 精简版M3U播放列表
- `ai_model.pkl`: AI模型文件
- `training_data.pkl`: 训练数据文件
- `iptv_processor.log`: 运行日志

## 常见问题

### 1. 程序运行时提示缺少模块

请确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

### 2. AI模型加载失败

尝试重新训练模型：
```bash
python train_model.py
```

### 3. 频道分类不准确

可以通过以下方式改进：
1. 编辑 `assets/corrections_name.txt` 添加纠错规则
2. 编辑 `category_config.json` 调整分类规则
3. 编辑 `manual_training.csv` 添加训练样本并重新训练模型

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至：your.email@example.com
