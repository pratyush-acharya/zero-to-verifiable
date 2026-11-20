"""The typed configs plus the helper's plumbing must equal the validated configs. Needs torch/transformers/trl/peft
(CPU is enough, nothing is downloaded); skipped otherwise."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

import torch
import transformers.training_args as ta
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen2Config, TrainingArguments
from trl import GRPOConfig

import z2v.train as zt

RUNS, SEED = zt.RUNS, zt.SEED


@pytest.fixture(params=[True, False], ids=["bf16", "fp16"])
def bf16(request, monkeypatch):
    "Both precisions, whatever this machine's GPU (the configs only check that bf16 exists)."
    monkeypatch.setattr(ta, "is_torch_bf16_gpu_available", lambda: True)
    monkeypatch.setattr(zt, "precision", lambda: dict(bf16=request.param, fp16=not request.param))
    return request.param


def same(a, b):
    "The names of the fields that differ between two configs."
    da, db = a.to_dict(), b.to_dict()
    return {k for k in da.keys() | db.keys() if da.get(k) != db.get(k)}


def test_sft_config_equals_the_validated_one(bf16):
    typed = TrainingArguments(learning_rate=2e-4, per_device_train_batch_size=32, lr_scheduler_type="linear", warmup_steps=20)
    validated = TrainingArguments(
        output_dir=f"{RUNS}/sft", per_device_train_batch_size=32, max_steps=378,
        learning_rate=2e-4, lr_scheduler_type="linear", warmup_steps=20, logging_steps=5,
        report_to=["tensorboard"], bf16=bf16, fp16=not bf16, save_strategy="no", remove_unused_columns=False, seed=SEED)
    assert same(zt.sft_args(typed, max_steps=378), validated) == set()


def test_grpo_config_equals_the_validated_one(bf16):
    typed = GRPOConfig(num_generations=8, temperature=1.0, learning_rate=2e-5, beta=0.0,
                       max_completion_length=64, per_device_train_batch_size=32, gradient_accumulation_steps=2)
    validated = GRPOConfig(
        output_dir=f"{RUNS}/grpo", per_device_train_batch_size=32, gradient_accumulation_steps=2,
        num_generations=8, max_completion_length=64, temperature=1.0, beta=0.0, scale_rewards="group",
        learning_rate=2e-5, lr_scheduler_type="constant_with_warmup", warmup_steps=10, max_grad_norm=0.1,
        max_steps=1000, logging_steps=1, report_to=["tensorboard"], bf16=bf16, fp16=not bf16,
        gradient_checkpointing=False, save_strategy="no", seed=SEED)
    merged = zt.grpo_args(typed)
    assert same(merged, validated) == set()
    assert (merged.generation_batch_size, merged.steps_per_generation) == (64, 2)


def test_knobs_you_set_win_over_the_validated_defaults(bf16):
    cfg = GRPOConfig(num_generations=8, per_device_train_batch_size=32, gradient_accumulation_steps=2,
                     lr_scheduler_type="cosine", warmup_steps=3, max_grad_norm=0.5, seed=7)
    merged = zt.grpo_args(cfg)
    assert (merged.lr_scheduler_type, merged.warmup_steps, merged.max_grad_norm, merged.seed) == ("cosine", 3, 0.5, 7)
    assert merged.report_to == ["tensorboard"] and merged.gradient_checkpointing is False
    sft = zt.sft_args(TrainingArguments(learning_rate=1e-4, logging_steps=7), max_steps=10)
    assert (sft.learning_rate, sft.logging_steps, sft.max_steps, sft.seed) == (1e-4, 7, 10, SEED)


def tiny_qwen():
    "A two-layer Qwen2 with random weights: the same module names as Qwen2.5-0.5B, nothing downloaded."
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                                        num_attention_heads=4, num_key_value_heads=2, vocab_size=1000,
                                                        tie_word_embeddings=True))


def test_all_linear_targets_the_validated_seven_projections():
    validated = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    typed = LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0)
    typed.task_type = typed.task_type or "CAUSAL_LM"          # what load_model() does
    torch.manual_seed(1); a = get_peft_model(tiny_qwen(), validated)
    torch.manual_seed(1); b = get_peft_model(tiny_qwen(), typed)
    assert type(a) is type(b)
    adapted = lambda m: sorted(n for n, mod in m.named_modules() if hasattr(mod, "lora_A") and len(mod.lora_A))
    assert adapted(a) == adapted(b) and len(adapted(b)) == 2 * 7
    assert {n.rsplit(".", 1)[-1] for n in adapted(b)} == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys() and all(torch.equal(pa[k], pb[k]) for k in pa)       # same LoRA init, same order
