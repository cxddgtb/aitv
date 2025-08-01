FROM python:3.10-slim

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制代码
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 入口点
CMD ["python", "main.py"]
