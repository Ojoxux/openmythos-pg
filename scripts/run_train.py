"""学習エントリーポイント。configs/model_200m.py を読み、outputs/ に書き出す。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # configs/ を import可能に


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-dir", default=str(ROOT / "outputs" / "data"))
    p.add_argument("--out-dir", default=str(ROOT / "outputs"))
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    from configs.model_200m import build_config
    from openmythos_pg.train import TrainHParams, train

    info_path = Path(args.tokens_dir) / "tokenize_info.json"
    info = json.loads(info_path.read_text())
    vocab_size = info["vocab_size"]
    print(f"vocab_size from tokenize_info: {vocab_size}")

    hp = TrainHParams(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accum,
        max_seq_len=args.seq_len,
        max_steps=args.max_steps,
        n_loops_train=args.n_loops,
        seed=args.seed,
    )

    train(
        model_config_builder=build_config,
        tokens_dir=args.tokens_dir,
        out_dir=args.out_dir,
        vocab_size=vocab_size,
        hp=hp,
    )


if __name__ == "__main__":
    main()
