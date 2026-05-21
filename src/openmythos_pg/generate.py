"""checkpoint を読んで生成する小ユーティリティ。"""
import json
import time
from pathlib import Path

import torch
from rich.console import Console


def generate(
    ckpt_path: Path | str,
    model_config_builder,
    tokenizer_name: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    n_loops: int = 4,
    out_path: Path | str | None = None,
) -> dict:
    """1プロンプトに対して generate を実行し、結果を返す/保存する。"""
    from open_mythos.main import OpenMythos
    from transformers import AutoTokenizer

    console = Console()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    cfg = model_config_builder(vocab_size=tokenizer.vocab_size)
    model = OpenMythos(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    console.log(f"prompt ids shape: {prompt_ids.shape}, n_loops={n_loops}")

    t0 = time.time()
    with torch.no_grad():
        # open_mythos の OpenMythos.generate は greedy/sampling 両対応の可能性。
        # 引数体系が不明な場合は自前のサンプリングループを使う (下記)。
        out_ids = _sample(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n_loops=n_loops,
        )
    dt = time.time() - t0
    out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    console.log(f"[green]generated[/] in {dt:.2f}s")
    console.print(out_text)

    record = {
        "ckpt": str(ckpt_path),
        "step": int(ckpt.get("step", -1)),
        "prompt": prompt,
        "output": out_text,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n_loops": n_loops,
        "elapsed_sec": dt,
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps(record, ensure_ascii=False, indent=2)
        )
    return record


def _sample(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    n_loops: int,
) -> torch.Tensor:
    """自前サンプリングループ (top-p + temperature)。"""
    ids = prompt_ids
    for _ in range(max_new_tokens):
        # context window で切る
        ctx = ids[:, -1024:]
        if ids.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(ctx, n_loops=n_loops)
        else:
            logits = model(ctx, n_loops=n_loops)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        # top-p
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        mask[..., 0] = False  # keep at least one
        sorted_logits[mask] = float("-inf")
        # unsort
        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered.scatter_(1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits_filtered, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids
