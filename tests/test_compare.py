"""results(), room_for_rl() and operator_mix() on simulated runs built to have the validated run's headline numbers."""
import re

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from z2v.compare import Run, arms_table, live_numbers, operator_mix, results, room_for_rl
from z2v.puzzles import make_puzzles

LIVE = "base=3.0&sft=36.3&rl=57.0&avg_sft=23.3&avg_rl=56.8&any_sft=69.3&any_rl=63.3&rand=5.2&four_sft=26.0&four_rl=49.3"


def simulated(name, solved3, solved4, sample_hits, any_hits):
    "A Run whose test numbers are fixed: solved3/solved4 greedy, and k=8 samples with these totals."
    p = make_puzzles(); items = p.eval
    exact = np.zeros(len(items), int)
    exact[:solved3] = 1; exact[150:150 + solved4] = 1          # test = 150 three-number puzzles, then 150 four-number
    exact[300:310] = 1                                          # a few of bigger/seen/rl-seen
    hits = np.zeros((300, 8), int)
    left = sample_hits
    for i in range(any_hits): hits[i, 0] = 1; left -= 1
    i = 0
    while left:
        row = hits[i % any_hits]; j = row.sum()
        if j < 8: row[j] = 1; left -= 1
        i += 1
    outs = [r["answer"] if e else "I am not sure" for r, e in zip(items, exact)]
    return Run(name, items, 300, outs, exact, exact.copy(), hits)


@pytest.fixture(scope="module")
def runs():
    return (simulated("base", 9, 0, 24, 21), simulated("SFT", 70, 39, 560, 208), simulated("SFT→RL", 97, 74, 1363, 190))


def test_live_numbers_have_the_slides_format(runs):
    _, tab = arms_table(*runs)
    assert live_numbers(tab) == LIVE
    assert re.fullmatch(r"(\w+=\d+\.\d&){9}\w+=\d+\.\d", LIVE)


def test_random_path_row_matches_the_validated_table(runs):
    _, tab = arms_table(*runs)
    row = tab.loc["random legal path"].round(1).tolist()
    assert row == [5.2, 4.8, 5.7, 100.0, 7.9, 2.5, 6.3, 4.1, 3.3, 5.2, 31.6]     # the validated run's first row
    assert tab.loc["SFT→RL", ["greedy", "3 numbers", "4 numbers"]].round(1).tolist() == [57.0, 64.7, 49.3]


def test_results_prints_table_deltas_sentence_and_live_line(runs, capsys, tmp_path):
    results(*runs, csv=str(tmp_path / "results.csv"))
    out = capsys.readouterr().out
    assert "test = 300 new puzzles (150 with 3 numbers, 150 with 4)" in out
    assert f"live numbers for the slides: {LIVE}\n" in out
    for label in ("greedy", "avg of 8", "any of 8"):
        assert re.search(rf"SFT→RL − SFT   {label}\s+[+-]\d+\.\d pp   95% CI \[", out)
    assert re.search(r"SFT − base  greedy\s+\+33\.3 pp", out)
    assert "The sentence these numbers allow:" in out and 'Meanwhile "any of 8 right" moved by -6.0 pp' in out
    assert "newly solved" in out
    assert (tmp_path / "results.csv").read_text().splitlines()[0] == "random legal path,base,SFT,SFT→RL,split"


def test_room_for_rl(runs, capsys):
    room_for_rl(runs[0], runs[1])
    out = capsys.readouterr().out
    assert "SFT     greedy  36.3%   any of 8 right  69.3%" in out and "base    greedy   3.0%   any of 8 right   7.0%" in out


def test_operator_mix(runs, capsys):
    operator_mix(runs[1], runs[2])
    out = capsys.readouterr().out
    assert "worked solutions" in out and "SFT→RL" in out and "only + and −" in out
