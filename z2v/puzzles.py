"""Countdown puzzles: the rules, the generator, an exhaustive solver, and the floor a model has to beat.

Pure Python (numpy only for medians). Seed 2026 gives exactly the puzzles of the validated run.
"""
import random
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

CANON = {"x": "*", "×": "*", "÷": "/"}                                 # other ways to write * and /
STEP = re.compile(r"(\d+)\s*([-+*/x×÷])\s*(\d+)\s*=\s*(-?\d+)")        # one step of an answer: "a op b = c"


def apply(a, op, b):
    "Countdown rules: the result must be a positive whole number. None means the step is illegal."
    op = CANON.get(op, op)
    c = {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b and a % b == 0 else None}[op]
    return c if c is not None and c > 0 else None


def random_solution(nums, rng):
    "Combine the numbers in a random legal order; returns the steps and the final value."
    pool, steps = list(nums), []
    while len(pool) > 1:
        i, j = rng.sample(range(len(pool)), 2); op = rng.choice("+-*/")
        c = apply(pool[i], op, pool[j])
        if c is None: return None
        steps.append(f"{pool[i]}{op}{pool[j]}={c}")
        pool = [x for k, x in enumerate(pool) if k not in (i, j)] + [c]
    return steps, pool[0]


def make_puzzle(rng, n, hi):
    "n numbers from 1..hi; the end of a random legal path is the target if it lands in 10..100."
    while True:
        nums = [rng.randint(1, hi) for _ in range(n)]
        sol = random_solution(nums, rng)
        if sol and 10 <= sol[1] <= 100 and sol[1] not in nums:
            return {"nums": nums, "target": sol[1], "steps": sol[0]}


def derivations(pool, steps=()):
    "Every legal way to combine all the numbers: yields (final value, steps)."
    if len(pool) == 1: yield pool[0], steps; return
    for i in range(len(pool)):
        for j in range(len(pool)):
            for op in "+-*/":
                if i == j or (op in "+*" and i > j): continue      # a+b and b+a count once
                c = apply(pool[i], op, pool[j])
                if c is None: continue
                rest = [x for k, x in enumerate(pool) if k not in (i, j)]
                yield from derivations(rest + [c], steps + (f"{pool[i]}{op}{pool[j]}={c}",))


def make_row(split, p):
    "A puzzle as the notebook uses it; evaluation rows also get the random-path floor and their solution count."
    r = {"split": split, "n": len(p["nums"]), "nums": p["nums"], "target": p["target"],
         "prompt": f"Numbers: {' '.join(map(str, p['nums']))}\nTarget: {p['target']}", "answer": "\n".join(p["steps"])}
    if split in ("test", "bigger", "seen", "rl-seen"):          # the floor: chance that a random legal path hits
        finals = [v for v, _ in derivations(p["nums"])]
        r["p_random"], r["n_solutions"] = sum(v == p["target"] for v in finals) / len(finals), finals.count(p["target"])
    return r


SPLITS = {"test": (150, 150, 20), "dev": (32, 32, 20), "rl": (1000, 3000, 20), "train": (5000, 11000, 20),
          "bigger": (30, 30, 50)}                    # (3-number puzzles, 4-number puzzles, largest number)


@dataclass(repr=False)
class Puzzles:
    "All the splits. Each puzzle is a dict with nums, target, prompt and one worked solution ('answer')."
    train: list      # 16,000 worked examples for SFT
    rl: list         # 4,000 prompts for GRPO, scored only by the checker
    dev: list        # 64 puzzles for the live accuracy curves
    test: list       # 300 new puzzles for the final table
    bigger: list     # 60 puzzles with numbers up to 50
    seen: list       # 40 SFT puzzles, asked again after training
    rl_seen: list    # 40 RL puzzles, asked again after training

    @property
    def eval(self):
        "The puzzles every evaluation answers, in a fixed order: test first."
        return self.test + self.bigger + self.seen + self.rl_seen

    def __repr__(self):
        return "Puzzles(" + ", ".join(f"{k}={len(v)}" for k, v in vars(self).items()) + ")"


@lru_cache(maxsize=None)
def make_puzzles(seed=2026):
    "Every split from one seed; no (numbers, target) pair appears twice across splits."
    rng, used = random.Random(seed), set()

    def fresh(n, hi):
        while True:
            p = make_puzzle(rng, n, hi); key = (tuple(sorted(p["nums"])), p["target"])
            if key not in used: used.add(key); return p

    raw = {s: [fresh(3, hi) for _ in range(n3)] + [fresh(4, hi) for _ in range(n4)] for s, (n3, n4, hi) in SPLITS.items()}
    rows = {s: [make_row(s, p) for p in ps] for s, ps in raw.items()}
    seen = [make_row("seen", p) for p in random.Random(5).sample(raw["train"], 40)]
    rl_seen = [make_row("rl-seen", p) for p in random.Random(6).sample(raw["rl"], 40)]
    return Puzzles(**rows, seen=seen, rl_seen=rl_seen)


def show_puzzles(puzzles, k=4):
    "Split sizes and a few puzzles with their worked solutions."
    p = puzzles
    print(f"puzzles -> SFT {len(p.train)} | RL {len(p.rl)} | dev {len(p.dev)} | test {len(p.test)} | bigger {len(p.bigger)}")
    print("\nexample prompt:\n" + p.train[1]["prompt"] + "\n\nits worked solution:\n" + p.train[1]["answer"])
    print("\nmore:")
    for r in p.train[2:2 + k]:
        print(f"  {' '.join(map(str, r['nums'])):<12} -> {r['target']:<4} {r['answer'].replace(chr(10), ' ; ')}")


def floor_stats(puzzles):
    "For 3- and 4-number test puzzles: legal complete paths, solutions, and the chance a random legal path hits."
    stats = {}
    for n in (3, 4):
        t = [r for r in puzzles.test if r["n"] == n]
        paths = [sum(1 for _ in derivations(r["nums"])) for r in t]
        stats[n] = {"paths": int(np.median(paths)), "max_paths": max(paths),
                    "solutions": int(np.median([r["n_solutions"] for r in t])),
                    "p_random": 100 * np.mean([r["p_random"] for r in t])}
    return stats


def floor(puzzles):
    "Print what an exhaustive solver says about the test puzzles: the floor a model has to beat."
    for n, s in floor_stats(puzzles).items():
        print(f"test, {n} numbers: median {s['paths']:,} legal complete paths (max {s['max_paths']:,}), "
              f"median {s['solutions']} solutions | a random legal path hits {s['p_random']:.1f}%")
