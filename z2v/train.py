"""Training: SFT sized to a time budget, GRPO against your reward, and the callbacks that draw the dev curves.

Your config objects keep every knob you set. The helper copies them and adds plumbing (where to log, 16-bit
mode, no checkpoints), plus the validated value for any knob the notebook leaves at the library default.
"""
import math
import os
import random
import time
from dataclasses import fields, replace

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainerCallback

from .chat import msgs, tokenizer
from .measure import generate, score
from .model import OUT, RUNS, SEED, bf16, dtype
from .puzzles import make_puzzles
from .reference import check as rules

_writer = []     # one TensorBoard writer for the dev curves, created on first use (run "eval")


def precision():
    "16-bit training: bf16 where the GPU has it, fp16 otherwise."
    return dict(bf16=bf16(), fp16=not bf16())


def merge(cfg, always, unless_set):
    "Copy a config: `always` is applied as is; `unless_set` only where the knob is still at the library default."
    default = {f.name: f.default for f in fields(cfg)}
    fill = {k: v for k, v in unless_set.items() if getattr(cfg, k) == default[k]}
    return replace(cfg, **fill, **always)      # replace() builds a new config, so its own checks run again


def sft_args(cfg, max_steps):
    "Your SFT config plus plumbing: TensorBoard, 16-bit, no checkpoints, and max_steps from the timing probe."
    return merge(cfg, always=dict(output_dir=f"{RUNS}/sft", report_to=["tensorboard"], save_strategy="no",
                                  remove_unused_columns=False, max_steps=max_steps, **precision()),
                 unless_set=dict(logging_steps=5, seed=SEED))


def grpo_args(cfg):
    "Your GRPO config plus plumbing, and the validated values for knobs the notebook leaves at the default."
    return merge(cfg, always=dict(output_dir=f"{RUNS}/grpo", report_to=["tensorboard"], gradient_checkpointing=False,
                                  save_strategy="no", steps_per_generation=None, **precision()),
                 # steps_per_generation=None: TRL re-derives it from generation_batch_size (it refuses both being set)
                 unless_set=dict(scale_rewards="group", max_grad_norm=0.1, lr_scheduler_type="constant_with_warmup",
                                 warmup_steps=10, max_steps=1000, logging_steps=1, seed=SEED))


class Encoded(TorchDataset):
    "Rows encoded on the fly with your encode()."
    def __init__(self, rows, encode): self.rows, self.encode = rows, encode
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.encode(self.rows[i])


def collate(batch):
    "Right-pad a batch: pad tokens get label -100 and attention 0."
    mx = max(len(b["input_ids"]) for b in batch)
    ids_ = torch.full((len(batch), mx), tokenizer().pad_token_id, dtype=torch.long)
    lab = torch.full((len(batch), mx), -100, dtype=torch.long)
    am = torch.zeros((len(batch), mx), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids_[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        am[i, :n] = 1
    return {"input_ids": ids_, "labels": lab, "attention_mask": am}


class DevCurve(TrainerCallback):
    "Greedy solve rate on the dev puzzles at the start and every `every` steps: the accuracy curve, not just the loss."
    def __init__(self, model, key, items, every): self.model, self.key, self.items, self.every = model, key, items, every
    def log(self, step):
        if not _writer: _writer.append(SummaryWriter(f"{RUNS}/eval"))
        acc = 100 * np.mean(score(self.items, generate(self.model, self.items), rules)[0])
        _writer[0].add_scalar(self.key, acc, step); _writer[0].flush()
        print(f"  [step {step}] {self.key} = {acc:.1f}%")
    def on_train_begin(self, args, state, control, **kw): self.log(state.global_step)
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.every == 0: self.log(state.global_step)


class StopAt(TrainerCallback):
    "Hard wall-clock stop: the agenda is the budget."
    def __init__(self, seconds): self.seconds, self.t0 = seconds, None
    def on_train_begin(self, args, state, control, **kw): self.t0 = time.time()
    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.t0 > self.seconds:
            control.should_training_stop = True
            print(f"  wall-clock stop at step {state.global_step}")


def seconds_per_batch(model, rows, encode, bs):
    "Time one real forward+backward pass on `bs` rows (after one untimed warm-up)."
    def step():
        b = collate([encode(r) for r in random.sample(rows, bs)])
        with torch.autocast("cuda", dtype=dtype()):
            model(**{k: v.cuda() for k, v in b.items()}).loss.backward()
    model.train()
    step(); torch.cuda.synchronize()                    # warm-up, not timed
    t0 = time.time(); step(); torch.cuda.synchronize(); s = time.time() - t0
    model.zero_grad(set_to_none=True)
    return s


def train_sft(model, rows, encode, cfg, minutes=4, dev=None):
    "Supervised fine-tuning on the worked solutions; max_steps comes from a timed step so the run fits `minutes`."
    from .reference import encode as reference_encode
    if encode(rows[0]) != reference_encode(rows[0]):
        print("note: your encode() differs from the reference (from z2v.reference import encode)")
    bs, ga = cfg.per_device_train_batch_size, cfg.gradient_accumulation_steps
    s_per_step = seconds_per_batch(model, rows, encode, bs) * ga
    steps_per_epoch = math.ceil(len(rows) / (bs * ga))
    max_steps = min(2 * steps_per_epoch, max(50, int(0.85 * minutes * 60 / s_per_step)))
    print(f"{s_per_step:.2f} s per optimizer step ({bs * ga} puzzles) | 1 epoch = {steps_per_epoch} steps"
          f" -> max_steps {max_steps} (~{max_steps * s_per_step / 60:.1f} min, {max_steps * bs * ga:,} puzzles)")

    os.environ["TENSORBOARD_LOGGING_DIR"] = f"{RUNS}/sft"         # read when the Trainer is built
    trainer = Trainer(model=model, args=sft_args(cfg, max_steps), train_dataset=Encoded(rows, encode),
                      data_collator=collate,
                      callbacks=[DevCurve(model, "eval/exact_match_sft", dev or make_puzzles().dev, every=25),
                                 StopAt(minutes * 60)])
    trainer.train()
    model.z2v_stage = "SFT"
    model.save_pretrained(f"{OUT}/adapter_sft"); tokenizer().save_pretrained(f"{OUT}/adapter_sft")
    print("saved:", f"{OUT}/adapter_sft")


def train_grpo(model, rows, reward, cfg, minutes=12, dev=None):
    "GRPO: sample a group of answers per puzzle, score them with `reward`, push up the better ones; stops at `minutes`."
    from datasets import Dataset
    from trl import GRPOTrainer
    data = Dataset.from_list([{"prompt": msgs(r), "nums": r["nums"], "target": r["target"]} for r in rows])
    print(f"RL puzzles: {len(data)}")
    args = grpo_args(cfg)

    def reward_countdown(*a, **kw):      # the validated run's name, so TensorBoard shows the same rewards/... tags
        return reward(*a, **kw)

    os.environ["TENSORBOARD_LOGGING_DIR"] = f"{RUNS}/grpo"
    trainer = GRPOTrainer(model=model, reward_funcs=reward_countdown, args=args, train_dataset=data,
                          processing_class=tokenizer(),
                          callbacks=[DevCurve(model, "eval/exact_match_rl", dev or make_puzzles().dev, every=25),
                                     StopAt(minutes * 60)])
    print(f"ready: {args.generation_batch_size // args.num_generations} puzzles x {args.num_generations} samples per step,"
          f" lr {args.learning_rate}, stops at {minutes} min")
    trainer.train()
    model.z2v_stage = "SFT→RL" if getattr(model, "z2v_stage", None) == "SFT" else "RL"
    model.save_pretrained(f"{OUT}/adapter_rl"); tokenizer().save_pretrained(f"{OUT}/adapter_rl")
