"""Comparing the arms: room for RL, the results table with bootstrap intervals, the live numbers, the operator mix.

Pure numpy/pandas/matplotlib: it only reads the scored answers that measure.evaluate() returns.
"""
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .puzzles import CANON, STEP


@dataclass(repr=False)
class Run:
    "One arm, scored: greedy answers on every eval puzzle and k samples on every test puzzle."
    name: str
    items: list           # the eval puzzles: test first, then bigger, seen, rl-seen
    n_test: int
    outs: list            # the greedy answer to each puzzle
    exact: np.ndarray     # 1 = legal and hits the target
    legal: np.ndarray     # 1 = every step true, every number used once
    hits: np.ndarray      # (test puzzles, k): 1 = that sample solved the puzzle

    def __repr__(self):
        return f"Run({self.name!r}: greedy {100 * self.exact[:self.n_test].mean():.1f}% on {self.n_test} test puzzles)"


def boot(x, B=10_000, seed=0):
    "Mean and 95% bootstrap interval over puzzles, in percent."
    x = np.asarray(x, float); m = x[np.random.default_rng(seed).integers(0, len(x), (B, len(x)))].mean(1)
    return 100*x.mean(), 100*np.percentile(m, 2.5), 100*np.percentile(m, 97.5)


def room_for_rl(*runs):
    "Greedy vs any-of-k on the test puzzles: RL can only reinforce answers the model sometimes samples."
    k = runs[0].hits.shape[1]
    print(f"room for RL on the {runs[0].n_test} test puzzles: one greedy answer vs any of {k} samples at temperature 1")
    for r in runs:
        print(f"  {r.name:<7} greedy {100 * r.exact[:r.n_test].mean():5.1f}%   any of {k} right {100 * r.hits.max(1).mean():5.1f}%")


def arms_table(base, sft, rl):
    "The four arms (random legal path, base, SFT, SFT→RL) as arrays, and the summary table in percent."
    items, nT, k = base.items, base.n_test, base.hits.shape[1]
    p_rand = np.array([r["p_random"] for r in items])
    arms = {"random legal path": dict(exact=p_rand, legal=np.ones(len(items)), sampled=p_rand[:nT],
                                      any8=1 - (1 - p_rand[:nT])**k)}
    for arm, run in [("base", base), ("SFT", sft), ("SFT→RL", rl)]:
        arms[arm] = dict(exact=np.array(run.exact, float), legal=np.array(run.legal, float),
                         sampled=run.hits.mean(1), any8=run.hits.max(1).astype(float))
    split = np.array([r["split"] for r in items]); n = np.array([r["n"] for r in items])
    rows = {}
    for arm, a in arms.items():
        e, lo, hi = boot(a["exact"][:nT])
        rows[arm] = {"greedy": e, "ci_lo": lo, "ci_hi": hi, "legal": 100*a["legal"][:nT].mean(),
                     "3 numbers": 100*a["exact"][(split == "test") & (n == 3)].mean(),
                     "4 numbers": 100*a["exact"][(split == "test") & (n == 4)].mean(),
                     "bigger": 100*a["exact"][split == "bigger"].mean(), "seen": 100*a["exact"][split == "seen"].mean(),
                     "rl puzzles": 100*a["exact"][split == "rl-seen"].mean(),
                     f"avg of {k}": 100*a["sampled"].mean(), f"any of {k}": 100*a["any8"].mean()}
    return arms, pd.DataFrame(rows).T


def live_numbers(tab):
    "The headline numbers as one line, in the format the slides read."
    k = [c for c in tab.columns if c.startswith("avg of")][0].split()[-1]
    t = tab.loc
    pairs = [("base", t["base", "greedy"]), ("sft", t["SFT", "greedy"]), ("rl", t["SFT→RL", "greedy"]),
             ("avg_sft", t["SFT", f"avg of {k}"]), ("avg_rl", t["SFT→RL", f"avg of {k}"]),
             ("any_sft", t["SFT", f"any of {k}"]), ("any_rl", t["SFT→RL", f"any of {k}"]),
             ("rand", t["random legal path", "greedy"]), ("four_sft", t["SFT", "4 numbers"]),
             ("four_rl", t["SFT→RL", "4 numbers"])]
    return "&".join(f"{key}={value:.1f}" for key, value in pairs)


def results(base, sft, rl, csv="/content/ckpt/results.csv"):
    "The final table with 95% intervals, the paired differences, the sentence they allow, the live numbers and a chart."
    import matplotlib.pyplot as plt
    arms, tab = arms_table(base, sft, rl)
    nT, k = base.n_test, base.hits.shape[1]
    sizes = Counter(r["n"] for r in base.items[:nT])
    print(f"test = {nT} new puzzles ({sizes[3]} with 3 numbers, {sizes[4]} with 4) | bigger = numbers up to 50\n")
    print(tab.round(1).to_string())
    if os.path.isdir(os.path.dirname(csv)):
        split = np.array([r["split"] for r in base.items])
        pd.DataFrame({arm: a["exact"] for arm, a in arms.items()}).assign(split=split).to_csv(csv, index=False)

    print()
    deltas = {}
    for a, b, what in [("SFT", "base", "exact"), ("SFT→RL", "SFT", "exact"), ("SFT→RL", "SFT", "sampled"), ("SFT→RL", "SFT", "any8")]:
        d, lo, hi = deltas[a, b, what] = boot(arms[a][what][:nT] - arms[b][what][:nT])     # paired: same puzzles
        label = {"exact": "greedy", "sampled": f"avg of {k}", "any8": f"any of {k}"}[what]
        print(f"{a:>6} − {b:<5} {label:<8} {d:+6.1f} pp   95% CI [{lo:+.1f}, {hi:+.1f}]   "
              f"{'distinguishable' if lo > 0 or hi < 0 else 'NOT distinguishable'} from zero")
    s, r = arms["SFT"]["exact"][:nT], arms["SFT→RL"]["exact"][:nT]
    print(f"RL vs SFT, puzzle by puzzle (greedy): {int(((r == 1) & (s == 0)).sum())} newly solved, {int(((r == 0) & (s == 1)).sum())} lost")

    d, lo, hi = deltas["SFT→RL", "SFT", "exact"]
    a8 = deltas["SFT→RL", "SFT", "any8"]
    explore = ("it commits more and explores less" if a8[2] < 0 else "it also finds more puzzles it can solve" if a8[1] > 0
               else "the change is not distinguishable from zero")
    print("\nThe sentence these numbers allow:")
    print(f'  "On {nT} new Countdown puzzles, a random legal path hits the target {tab.loc["random legal path", "greedy"]:.0f}% of the time '
          f'and the base model solved {tab.loc["base", "greedy"]:.0f}%. After SFT on one worked solution per puzzle it solved '
          f'{tab.loc["SFT", "greedy"]:.0f}%, and at least one of {k} samples was right on {tab.loc["SFT", f"any of {k}"]:.0f}%. '
          f'GRPO, rewarded for any correct solution, changed greedy accuracy by {d:+.1f} pp (95% CI {lo:+.1f} to {hi:+.1f}), '
          f'which is {"" if lo > 0 or hi < 0 else "not "}distinguishable from zero. Meanwhile "any of {k} right" moved by '
          f'{a8[0]:+.1f} pp (95% CI {a8[1]:+.1f} to {a8[2]:+.1f}): {explore}."')
    print(f"\nlive numbers for the slides: {live_numbers(tab)}")

    fig, ax = plt.subplots(figsize=(5.6, 3.0), dpi=150)
    x = np.arange(len(tab)); w = 0.38
    ax.bar(x - w/2, tab["greedy"], w, color="#2b6cb0", label="greedy",
           yerr=[tab["greedy"] - tab["ci_lo"], tab["ci_hi"] - tab["greedy"]], capsize=3, ecolor="#222")
    ax.bar(x + w/2, tab[f"any of {k}"], w, color="#a0aec0", label=f"any of {k} samples")
    ax.set_xticks(x); ax.set_xticklabels(tab.index, fontsize=7); ax.set_ylim(0, 100); ax.set_ylabel("puzzles solved, %")
    ax.legend(fontsize=7, loc="upper left"); ax.set_title(f"{nT} new puzzles, 95% CI on greedy", fontsize=8)
    plt.tight_layout(); plt.show()


def op_profile(texts, legal):
    "Operator mix over the legal answers: which moves does the model reach for?"
    used = [[CANON.get(op, op) for _, op, _, _ in STEP.findall(t)] for t, ok in zip(texts, legal) if ok]
    ops = Counter(op for u in used for op in u); total = sum(ops.values()) or 1
    return {**{f"{op} share": 100 * ops[op] / total for op in "+-*/"},
            "only + and −": 100 * np.mean([set(u) <= {"+", "-"} for u in used]) if used else 0.0,
            "legal answers": len(used)}


def operator_mix(*runs):
    "Operators in the worked solutions SFT imitated vs in each run's legal answers on the test puzzles."
    from .reference import check
    test = runs[0].items[:runs[0].n_test]
    worked = [r["answer"] for r in test]
    prof = {"worked solutions": op_profile(worked, [check(t, r["nums"])[0] for t, r in zip(worked, test)])}
    for run in runs:
        prof[run.name] = op_profile(run.outs[:run.n_test], run.legal[:run.n_test])
    print(f"operators in legal answers on the {len(test)} test puzzles (shares in %)\n")
    print(pd.DataFrame(prof).T.round(1).to_string())
