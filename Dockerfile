# 使用官方的 Python 3.9 镜像作为基础
FROM python:3.9-slim

# 安装 supervisord
RUN apt-get update && apt-get install -y supervisor

# 创建必要的目录
RUN mkdir -p /content/data /chroma_db

# 设置工作目录
WORKDIR /app

# 复制应用程序代码到容器中
COPY . /app

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 复制 supervisord 配置文件
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 暴露应用程序使用的端口
EXPOSE 7860

# 设置环境变量
ENV PDF_FOLDER=/content/data
ENV CHROMA_DB_DIR=/chroma_db

# 启动 supervisord
CMD ["supervisord", "-n"]