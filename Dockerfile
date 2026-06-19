# ============================================================
# Synapse - 多 Agent 知识检索平台 Dockerfile
# ============================================================
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/

# 注意：.env 不打包进镜像（含 API 密钥，入镜像层会泄漏）。
# 运行时配置由 docker-compose 的 env_file 注入为环境变量，
# app/config.py 直接读取环境变量，无需容器内 .env 文件。

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
