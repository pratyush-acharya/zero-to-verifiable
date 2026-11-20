"""The puzzles must be exactly the validated run's puzzles, and the solver facts must match what the slides quote."""
import random
import subprocess
import sys

from z2v.puzzles import apply, derivations, floor, floor_stats, make_puzzles, show_puzzles


def test_make_puzzles_reproduces_the_validated_puzzles(validated_data):
    p, v = make_puzzles(), validated_data
    for split in ("train", "rl", "dev", "test", "bigger"):
        mine, theirs = getattr(p, split), v["DATA"][split]
        assert len(mine) == len(theirs), split
        for i, (a, b) in enumerate(zip(mine, theirs)):
            assert a == b, f"{split}[{i}]: {a} != {b}"
    assert p.seen == v["SEEN"] and p.rl_seen == v["RL_SEEN"]
    assert p.eval == v["TEST"] + v["BIGGER"] + v["SEEN"] + v["RL_SEEN"]      # the validated EVAL order


def test_split_sizes_and_uniqueness():
    p = make_puzzles()
    assert [len(p.train), len(p.rl), len(p.dev), len(p.test), len(p.bigger)] == [16000, 4000, 64, 300, 60]
    keys = [(tuple(sorted(r["nums"])), r["target"]) for s in (p.train, p.rl, p.dev, p.test, p.bigger) for r in s]
    assert len(keys) == len(set(keys))


def test_rules_match_the_validated_rules(validated_data):
    v, rng = validated_data, random.Random(0)
    for _ in range(5000):
        a, b, op = rng.randint(0, 60), rng.randint(0, 60), rng.choice("+-*/")
        assert apply(a, op, b) == v["apply"](a, op, b)
    for nums in ([14, 18, 3], [12, 1, 4, 8], [1, 1, 1, 1], [20, 19, 17, 13], [50, 2, 7]):
        assert list(derivations(nums)) == list(v["derivations"](nums))


def test_floor_matches_the_numbers_the_slides_quote(capsys):
    s = floor_stats(make_puzzles())
    assert (s[3]["paths"], s[4]["paths"], s[4]["max_paths"]) == (31, 644, 1016)
    assert (s[3]["solutions"], s[4]["solutions"]) == (3, 18)
    assert (round(s[3]["p_random"], 1), round(s[4]["p_random"], 1)) == (7.9, 2.5)
    floor(make_puzzles())
    out = capsys.readouterr().out
    assert "median 644 legal complete paths (max 1,016)" in out and "hits 2.5%" in out and "hits 7.9%" in out


def test_show_puzzles_prints_the_worked_example(capsys):
    show_puzzles(make_puzzles())
    out = capsys.readouterr().out
    assert "puzzles -> SFT 16000 | RL 4000 | dev 64 | test 300 | bigger 60" in out
    assert "Numbers: 14 18 3\nTarget: 68" in out and "3*18=54\n14+54=68" in out


def test_pure_parts_import_without_torch():
    code = ("import sys, z2v, z2v.puzzles, z2v.chat, z2v.reference, z2v.compare, z2v.explain; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=__file__.rsplit("/tests/", 1)[0])
