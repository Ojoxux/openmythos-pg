# uv が詰まった人向けのおまけ。pytorch/pytorch ベースで CUDA12.1 を載せる。
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# uvをインストール
RUN pip install --no-cache-dir uv

WORKDIR /workspace
COPY pyproject.toml /workspace/

# 依存だけ先に解決 (CUDA torch は base image のものを優先)
RUN uv pip install --system --no-build-isolation \
      "open-mythos==0.5.0" \
      "transformers>=4.40.0" \
      "datasets>=2.18.0" \
      "tokenizers>=0.15.0" \
      "sentencepiece" \
      "protobuf" \
      "numpy<2.0" \
      "rich" \
      "matplotlib"

# コードをコピー
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY configs/ /workspace/configs/

ENV PYTHONPATH=/workspace/src:/workspace

CMD ["python", "scripts/run_train.py", "--help"]
