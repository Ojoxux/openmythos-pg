"""200M OpenMythos の学習ループ (plain PyTorch + bf16)."""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from openmythos_pg.data import load_memmap, sample_batch
from openmythos_pg.logging_utils import JsonlLogger, plot_loss_curve


@dataclass
class TrainHParams:
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 2
    grad_accumulation: int = 8
    max_seq_len: int = 1024
    max_steps: int = 50_000
    eval_interval: int = 1000
    save_interval: int = 5000
    eval_iters: int = 50
    n_loops_train: int = 4
    seed: int = 1337
    keep_last_ckpts: int = 3


def cosine_lr(step: int, hp: TrainHParams) -> float:
    """warmup → cosine decay。"""
    if step < hp.warmup_steps:
        return hp.learning_rate * step / hp.warmup_steps
    progress = (step - hp.warmup_steps) / max(1, hp.max_steps - hp.warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return hp.min_lr + coeff * (hp.learning_rate - hp.min_lr)


def estimate_loss(
    model, train_mm, val_mm, hp: TrainHParams, device, rng
) -> dict[str, float]:
    """train/val それぞれ eval_iters バッチでの平均 loss を返す。"""
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for split, mm in [("train", train_mm), ("val", val_mm)]:
            running = 0.0
            for _ in range(hp.eval_iters):
                x, y = sample_batch(mm, hp.batch_size, hp.max_seq_len, rng)
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x, n_loops=hp.n_loops_train)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    )
                running += loss.item()
            losses[split] = running / hp.eval_iters
    model.train()
    return losses


def save_checkpoint(
    out_dir: Path, step: int, model, optimizer, hp: TrainHParams, keep: int
):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hparams": asdict(hp),
        },
        path,
    )
    # rotate
    ckpts = sorted(out_dir.glob("step_*.pt"))
    for old in ckpts[:-keep]:
        old.unlink()


def train(
    model_config_builder,
    tokens_dir: Path | str,
    out_dir: Path | str,
    vocab_size: int,
    hp: TrainHParams = TrainHParams(),
) -> None:
    """学習エントリーポイント。

    Args:
        model_config_builder: vocab_size → MythosConfig を返す関数
        tokens_dir: train.bin / val.bin が置いてあるディレクトリ
        out_dir: checkpoints / logs を吐き出すディレクトリ
        vocab_size: tokenizer の vocab サイズ
        hp: TrainHParams
    """
    from open_mythos.main import OpenMythos

    console = Console()
    out_dir = Path(out_dir)
    tokens_dir = Path(tokens_dir)
    logs_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- system / config dump ---
    sys_info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "cuda_capability": (
            torch.cuda.get_device_capability(0)
            if torch.cuda.is_available()
            else None
        ),
    }
    (out_dir / "system_info.json").write_text(json.dumps(sys_info, indent=2))
    (out_dir / "config_used.json").write_text(
        json.dumps({"hparams": asdict(hp), "vocab_size": vocab_size}, indent=2)
    )
    console.log("[bold cyan]system_info:[/]", sys_info)

    # --- data ---
    train_mm = load_memmap(tokens_dir / "train.bin")
    val_mm = load_memmap(tokens_dir / "val.bin")
    console.log(f"train tokens: {len(train_mm):,}  val tokens: {len(val_mm):,}")

    # --- model ---
    cfg = model_config_builder(vocab_size=vocab_size)
    model = OpenMythos(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    console.log(f"[bold green]params:[/] {n_params:,} ({n_params / 1e6:.1f}M)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # --- optimizer (embedding を weight_decay から除外) ---
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embed" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optim_groups = [
        {"params": decay, "weight_decay": hp.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=hp.learning_rate, betas=(hp.beta1, hp.beta2)
    )

    # --- training ---
    rng = np.random.default_rng(hp.seed)
    logger = JsonlLogger(logs_dir / "train.jsonl")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("step {task.completed}/{task.total}"),
        TextColumn("loss={task.fields[loss]:.3f}"),
        TextColumn("lr={task.fields[lr]:.2e}"),
        TextColumn("{task.fields[tps]} tok/s"),
        TimeRemainingColumn(),
        console=console,
    )

    last_log_t = time.time()
    last_log_step = 0
    with progress:
        task = progress.add_task(
            "train", total=hp.max_steps, loss=0.0, lr=0.0, tps="—"
        )
        for step in range(1, hp.max_steps + 1):
            lr = cosine_lr(step, hp)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # gradient accumulation
            for micro in range(hp.grad_accumulation):
                x, y = sample_batch(train_mm, hp.batch_size, hp.max_seq_len, rng)
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x, n_loops=hp.n_loops_train)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    )
                    loss = loss / hp.grad_accumulation
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # log every 10 steps
            if step % 10 == 0 or step == 1:
                now = time.time()
                dt = now - last_log_t
                tokens = (
                    (step - last_log_step)
                    * hp.batch_size
                    * hp.grad_accumulation
                    * hp.max_seq_len
                )
                tps = int(tokens / dt) if dt > 0 else 0
                logger.log(
                    {
                        "step": step,
                        "train_loss": float(loss.item() * hp.grad_accumulation),
                        "val_loss": None,
                        "lr": lr,
                        "tokens_per_sec": tps,
                        "vram_mb": (
                            int(torch.cuda.memory_allocated() / 1e6)
                            if device == "cuda"
                            else 0
                        ),
                    }
                )
                progress.update(
                    task,
                    completed=step,
                    loss=float(loss.item() * hp.grad_accumulation),
                    lr=lr,
                    tps=f"{tps:,}",
                )
                last_log_t = now
                last_log_step = step

            # eval
            if step % hp.eval_interval == 0:
                losses = estimate_loss(model, train_mm, val_mm, hp, device, rng)
                logger.log(
                    {
                        "step": step,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "lr": lr,
                    }
                )
                console.log(
                    f"[yellow]eval[/] step={step} "
                    f"train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}"
                )

            # save
            if step % hp.save_interval == 0 or step == hp.max_steps:
                save_checkpoint(
                    ckpt_dir, step, model, optimizer, hp, hp.keep_last_ckpts
                )
                console.log(f"[green]saved[/] checkpoint at step {step}")

    logger.close()
    # finalize: loss curve + vram peak
    plot_loss_curve(logs_dir / "train.jsonl", logs_dir / "loss_curve.png")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e6
        (out_dir / "vram_peak.txt").write_text(f"{peak:.1f} MB\n")
        console.log(f"[bold red]VRAM peak:[/] {peak:.1f} MB")
    console.log("[bold green]training complete.[/]")
