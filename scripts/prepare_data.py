"""Wikipedia-ja を tokenize して outputs/data/{train,val}.bin に保存する。

Win側で初回1回だけ実行。HFキャッシュも作られる (~3.5GB)。
"""
import argparse
import json
import sys
from pathlib import Path

# src/ を import path に追加
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(ROOT / "outputs" / "data"))
    p.add_argument("--tokenizer", default="llm-jp/llm-jp-3-1.8b")
    p.add_argument("--dataset", default="wikimedia/wikipedia")
    p.add_argument("--dataset-config", default="20231101.ja")
    p.add_argument("--val-ratio", type=float, default=0.005)
    args = p.parse_args()

    from openmythos_pg.data import tokenize_corpus

    info = tokenize_corpus(
        out_dir=args.out_dir,
        tokenizer_name=args.tokenizer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        val_ratio=args.val_ratio,
    )
    info_path = Path(args.out_dir) / "tokenize_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"\nwrote {info_path}")


if __name__ == "__main__":
    main()
