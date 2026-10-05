# ============================================================
# Synapse - 全能智能助手平台 Dockerfile
# ============================================================
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（git：代码仓库助手的本地仓库 / clone 功能）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
# INSTALL_EXTRAS=0 可跳过可选能力（Claude / MCP / PostgreSQL / PDF·DOCX），构建更精简的镜像
COPY requirements.txt requirements-extras.txt ./
ARG INSTALL_EXTRAS=1
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_EXTRAS" = "1" ]; then pip install --no-cache-dir -r requirements-extras.txt; fi

# 复制应用代码与内置插件
COPY app/ ./app/
COPY plugins/ ./plugins/

# 数据目录（SQLite、上传文件、仓库 clone、密钥文件），运行时由 docker-compose 挂载卷
RUN mkdir -p /app/data

# 注意：.env 不打包进镜像（含 API 密钥，入镜像层会泄漏）。
# 运行时配置由 docker-compose 的 env_file 注入为环境变量，
# app/config.py 直接读取环境变量，无需容器内 .env 文件。

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令（单 worker：定时任务调度器与异常检测状态在进程内）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
