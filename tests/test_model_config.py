"""configs/model_200m.py の単体テスト。GPU不要。"""
import pytest


def test_build_config_returns_mythos_config():
    from open_mythos.main import MythosConfig
    from configs.model_200m import build_config

    cfg = build_config(vocab_size=96867)

    assert isinstance(cfg, MythosConfig)
    assert cfg.vocab_size == 96867
    assert cfg.dim == 768
    assert cfg.n_heads == 12
    assert cfg.max_seq_len == 1024
    assert cfg.attn_type == "mla"


def test_build_config_mla_params_consistent():
    from configs.model_200m import build_config

    cfg = build_config(vocab_size=1000)

    # MLA は kv_lora_rank, q_lora_rank などのフィールドが必要
    assert cfg.kv_lora_rank == 128
    assert cfg.q_lora_rank == 256
    assert cfg.qk_rope_head_dim == 32
    assert cfg.qk_nope_head_dim == 32
    assert cfg.v_head_dim == 64
    assert cfg.n_kv_heads == 12


def test_build_config_moe_params():
    from configs.model_200m import build_config

    cfg = build_config(vocab_size=1000)
    assert cfg.n_experts == 4
    assert cfg.n_shared_experts == 1
    assert cfg.n_experts_per_tok == 2
    assert cfg.expert_dim == 2048
