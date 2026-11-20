"""Measuring: generate answers (greedy, or sampled the way GRPO explores), score them with a checker, print the report."""
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from .chat import render, tokenizer
from .compare import Run
from .model import dtype


@torch.inference_mode()
def generate(model, items, k=1, max_new_tokens=96, temperature=1.0):
    "Greedy answers when k == 1; otherwise k samples per puzzle at `temperature`, the way GRPO explores."
    tok = tokenizer()
    gen_kw = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.0, use_cache=True, pad_token_id=tok.pad_token_id,
                  eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")])
    # Passed explicitly: Qwen's shipped generation_config samples at T=0.7 with top-p/top-k and adds repetition_penalty=1.1.
    was_training = model.training; model.eval()
    mode = dict(do_sample=False) if k == 1 else dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, num_return_sequences=k)
    batch, outs = 64 if k == 1 else 32, []
    for i in range(0, len(items), batch):
        chunk = items[i:i+batch]
        enc = tok([render(r) for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.autocast("cuda", dtype=dtype()):
            gen = model.generate(**enc, **gen_kw, **mode)
        texts = [s.strip() for s in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        outs += texts if k == 1 else [texts[j*k:(j+1)*k] for j in range(len(chunk))]
    if was_training: model.train()
    return outs


def solved(r, text, check):
    "1 if the answer is legal and ends at the target."
    legal, final = check(text, r["nums"])
    return int(legal and final == r["target"])


def score(items, outs, check):
    "Per puzzle: solved (0/1) and legal steps (0/1)."
    return [solved(r, o, check) for r, o in zip(items, outs)], [int(check(o, r["nums"])[0]) for r, o in zip(items, outs)]


def check_the_checker(check, rows):
    "Stop early, with a clear message, if a typed check() disagrees with the rules."
    from .reference import check as rules
    for r in rows:
        steps = r["answer"].splitlines(); lhs, rhs = steps[0].split("="); a = r["nums"][0]
        for text in [r["answer"], "\n".join([f"{lhs}={int(rhs) + 1}"] + steps[1:]),   # worked solution, false arithmetic
                     "\n".join(steps[:-1]), f"{a}+{a}={2 * a}", ""]:                # a number left out, reused, no steps
            got, want = tuple(check(text, r["nums"])), rules(text, r["nums"])
            if got != want:
                raise ValueError(f"check({text!r}, {r['nums']}) returned {got}; the rules say {want}.\n"
                                 "Fix your check(), or carry on with the reference version:  from z2v.reference import check")


def print_splits(name, items, exact, legal):
    "One line per split: greedy solve rate and legal-steps rate."
    df = pd.DataFrame({"split": [r["split"] for r in items], "n": [r["n"] for r in items], "exact": exact, "legal": legal})
    for split, g in df.groupby("split", sort=False):
        extra = ("   (" + ", ".join(f"{n} numbers {100*h.exact.mean():.1f}%" for n, h in g.groupby("n")) + ")") if split == "test" else ""
        print(f"{name:<7} {split:<8} greedy solved {100*g.exact.mean():5.1f}%   legal steps {100*g.legal.mean():5.1f}%   n={len(g)}{extra}")


def show(items, outs, check, k=4):
    "The first few puzzles with the model's greedy answers."
    for r, o in zip(items[:k], outs[:k]):
        print(f"{'✓' if solved(r, o, check) else '✗'} {r['prompt'].replace(chr(10), ' | ')}  ->  {o[:90]!r}")


def evaluate(model, puzzles, check, adapter=True, k=8, name=None):
    "Greedy answers on every eval puzzle plus k samples at temperature 1 on each test puzzle, scored by `check`."
    check_the_checker(check, puzzles.test + puzzles.bigger)
    name = name or (getattr(model, "z2v_stage", "adapter") if adapter else "base")
    items, test = puzzles.eval, puzzles.test
    with (nullcontext() if adapter else model.disable_adapter()):       # adapter=False: the base weights only
        outs = generate(model, items)
        exact, legal = score(items, outs, check)
        print_splits(name, items, exact, legal)
        hits = np.array([[solved(r, o, check) for o in samples] for r, samples in zip(test, generate(model, test, k=k))])
        print(f"{name:<7} test, {k} samples each: average {100*hits.mean():5.1f}%   any of {k} right {100*hits.max(1).mean():5.1f}%")
    show(items, outs, check)
    return Run(name, items, len(test), outs, np.array(exact), np.array(legal), hits)
