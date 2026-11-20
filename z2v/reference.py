"""Reference versions of the three functions you type in the notebook.

If yours breaks and you want to keep going:   from z2v.reference import check, encode, reward
"""
from .chat import ids, render
from .puzzles import STEP, apply


def check(text, nums):
    "Replay the steps. Legal = every step is true arithmetic on numbers still available, and every number is used."
    pool, steps = list(nums), STEP.findall(text or "")
    if not steps: return False, None
    for a, op, b, c in steps:
        a, b, c = int(a), int(b), int(c)
        if a not in pool: return False, None
        pool.remove(a)
        if b not in pool: return False, None
        pool.remove(b)
        if apply(a, op, b) != c: return False, None      # apply() also reads x, × and ÷
        pool.append(c)
    return len(pool) == 1, (pool[0] if len(pool) == 1 else None)


def encode(p):
    "One training example: the full conversation's tokens, with label -100 (no loss) on everything before the answer."
    prompt_ids, full_ids = ids(render(p)), ids(render(p, p["answer"]))
    assert full_ids[:len(prompt_ids)] == prompt_ids, "the prompt tokens change once the answer is appended"
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def reward(completions, nums, target, **kw):
    "1.0 for a legal answer that hits the target, 0.5 * 0.9^|off| for a legal miss, 0 for anything else."
    scores = []
    for c, ns, t in zip(completions, nums, target):
        text = c if isinstance(c, str) else c[-1]["content"]      # GRPO passes chat messages; we want the text
        legal, final = check(text, ns)
        if not legal: scores.append(0.0)
        elif final == t: scores.append(1.0)
        else: scores.append(0.5 * 0.9 ** abs(final - t))
    return scores
