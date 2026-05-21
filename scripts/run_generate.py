"""checkpoint から複数プロンプト・複数 n_loops で生成。"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DEFAULT_PROMPTS = [
    "日本の首都は",
    "猫は",
    "昔々あるところに、",
    "def fibonacci(n):",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="llm-jp/llm-jp-3-1.8b")
    p.add_argument("--out-dir", default=str(ROOT / "outputs" / "samples"))
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument(
        "--n-loops-list",
        nargs="+",
        type=int,
        default=[4],
        help="複数指定で n_loops sweep",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()

    from configs.model_200m import build_config
    from openmythos_pg.generate import generate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    for prompt in args.prompts:
        for n_loops in args.n_loops_list:
            slug = (
                prompt.replace(" ", "_").replace("/", "_")[:20]
                + f"_loops{n_loops}_{ts}.json"
            )
            out_path = out_dir / slug
            print(f"\n=== prompt={prompt!r}  n_loops={n_loops} ===")
            generate(
                ckpt_path=args.ckpt,
                model_config_builder=build_config,
                tokenizer_name=args.tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                n_loops=n_loops,
                out_path=out_path,
            )


if __name__ == "__main__":
    main()
