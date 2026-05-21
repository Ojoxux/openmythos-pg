"""JSONL ログ + matplotlib loss曲線。W&B/TBは使わない方針。"""
import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    """1step1行のJSONログ。append modeで再開も安全。"""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", buffering=1)  # line buffered

    def log(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def plot_loss_curve(jsonl_path: Path | str, out_path: Path | str) -> None:
    """train.jsonl から train/val loss を抽出してPNG保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_train, train_loss = [], []
    steps_val, val_loss = [], []

    with Path(jsonl_path).open() as f:
        for line in f:
            r = json.loads(line)
            if "train_loss" in r and r["train_loss"] is not None:
                steps_train.append(r["step"])
                train_loss.append(r["train_loss"])
            if "val_loss" in r and r["val_loss"] is not None:
                steps_val.append(r["step"])
                val_loss.append(r["val_loss"])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    ax.plot(steps_train, train_loss, label="train", alpha=0.7, linewidth=1)
    if steps_val:
        ax.plot(steps_val, val_loss, label="val", marker="o", linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("OpenMythos training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
