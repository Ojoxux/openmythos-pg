"""200M parameter target OpenMythos config for RTX 3070 Laptop (8GB VRAM).

Param breakdown (estimated, vocab=96867):
  Embedding (tied):    ~74M
  Attention (MLA, x5): ~15M
  MoE FFN (x5):        ~80M
  Misc (LN, router):    ~5M
  Total:              ~175M
"""
from open_mythos.main import MythosConfig


def build_config(vocab_size: int) -> MythosConfig:
    """Build the 200M MLA + MoE config.

    Args:
        vocab_size: Tokenizer vocabulary size (e.g., 96867 for llm-jp-3).
    """
    return MythosConfig(
        # tokenizer-driven
        vocab_size=vocab_size,
        max_seq_len=1024,
        # backbone
        dim=768,
        n_heads=12,
        prelude_layers=2,
        coda_layers=2,
        max_loop_iters=4,
        # MoE FFN
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=2048,
        # MLA attention
        attn_type="mla",
        n_kv_heads=12,
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
    )
