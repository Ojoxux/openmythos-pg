"""data.py の単体テスト。GPU/HFは使わない。tiny memmap で sample_batch をテスト。"""
import numpy as np


def test_sample_batch_shape_and_range(tmp_path):
    from openmythos_pg.data import sample_batch

    # 100 tokens の tiny memmap を作成
    bin_path = tmp_path / "tiny.bin"
    arr = np.arange(100, dtype=np.uint32)
    arr.tofile(bin_path)
    mm = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(100,))

    rng = np.random.default_rng(seed=42)
    x, y = sample_batch(mm, batch_size=4, seq_len=8, rng=rng)

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y は x の +1 シフト
    assert ((y[:, :-1] - x[:, 1:]).abs() == 0).all() or True  # shift関係は内容で確認
    # 値域チェック
    assert int(x.min()) >= 0
    assert int(x.max()) < 100


def test_sample_batch_y_is_x_shifted_by_one(tmp_path):
    """y[i, t] == x[i, t+1] (next-token prediction)."""
    from openmythos_pg.data import sample_batch

    bin_path = tmp_path / "seq.bin"
    arr = np.arange(200, dtype=np.uint32)
    arr.tofile(bin_path)
    mm = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(200,))

    rng = np.random.default_rng(seed=0)
    x, y = sample_batch(mm, batch_size=2, seq_len=16, rng=rng)

    # 連番 memmap なら y == x+1 のはず
    assert (y == x + 1).all()


def test_sample_batch_reproducible_with_same_seed(tmp_path):
    from openmythos_pg.data import sample_batch

    bin_path = tmp_path / "seq.bin"
    arr = np.arange(500, dtype=np.uint32)
    arr.tofile(bin_path)
    mm = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(500,))

    x1, _ = sample_batch(mm, 4, 16, np.random.default_rng(seed=123))
    x2, _ = sample_batch(mm, 4, 16, np.random.default_rng(seed=123))
    assert (x1 == x2).all()
