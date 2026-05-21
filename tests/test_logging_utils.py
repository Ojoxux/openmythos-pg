"""logging_utils.py の単体テスト。GPU不要。"""
import json
import tempfile
from pathlib import Path


def test_jsonl_logger_writes_records():
    from openmythos_pg.logging_utils import JsonlLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "train.jsonl"
        logger = JsonlLogger(log_path)
        logger.log({"step": 1, "loss": 9.5, "lr": 3e-4})
        logger.log({"step": 2, "loss": 9.0, "lr": 3e-4})
        logger.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["step"] == 1
        assert record["loss"] == 9.5


def test_jsonl_logger_appends_when_reopened():
    from openmythos_pg.logging_utils import JsonlLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "train.jsonl"
        l1 = JsonlLogger(log_path)
        l1.log({"step": 1})
        l1.close()
        l2 = JsonlLogger(log_path)
        l2.log({"step": 2})
        l2.close()

        assert len(log_path.read_text().strip().split("\n")) == 2


def test_plot_loss_curve_creates_png(tmp_path):
    from openmythos_pg.logging_utils import plot_loss_curve

    jsonl_path = tmp_path / "train.jsonl"
    records = [
        {"step": i, "train_loss": 11.5 - i * 0.01, "val_loss": None}
        for i in range(100)
    ]
    # val 数件
    for i in range(0, 100, 10):
        records[i]["val_loss"] = 11.0 - i * 0.01
    with jsonl_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    out_path = tmp_path / "curve.png"
    plot_loss_curve(jsonl_path, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # 適当な最小サイズ
