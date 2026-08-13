FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 下载 spaCy 英文模型
RUN python -m spacy download en_core_web_sm

# 预下载 SentenceTransformer 模型，减少 Cloud Run Job 冷启动时间
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

CMD ["python", "main.py"]