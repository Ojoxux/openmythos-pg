# openmythos-pg

Windows + RTX 3070 Laptop (8GB) / Python 3.11 / CUDA 12.8 想定。

## uv

```powershell
winget install --id=astral-sh.uv

git clone https://github.com/Ojoxux/openmythos-pg
cd openmythos-pg

uv sync

uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`cuda: True` になること。`+cpu` なら:

```powershell
uv pip install "torch==2.11.0" --index https://download.pytorch.org/whl/cu128 --force-reinstall
```

## Docker (uv が詰まったら)

```bash
cd docker
docker compose build
docker compose up
```

WSL2 + Docker Desktop + NVIDIA Container Toolkit が必要そう
