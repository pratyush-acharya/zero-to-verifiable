"""check, encode and reward: the reference versions, the notebook's typed versions and the validated ones must agree."""
import random

import pytest
from conftest import FakeTokenizer, definitions, notebook_cell, validated_cell

import z2v.chat
from z2v import reference
from z2v.explain import ladder, reward_ladder
from z2v.puzzles import STEP, apply, make_puzzles


def typed(name, marker, **extra):
    "A function exactly as typed in notebooks/workshop.ipynb."
    ns = {"STEP": STEP, "apply": apply, "render": z2v.chat.render, "ids": z2v.chat.ids, **extra}
    exec(definitions(notebook_cell(marker), [name]), ns)
    return ns[name]


def texts_to_check():
    "Worked solutions, broken versions of them, and random step lists in every notation the regex reads."
    p, rng, out = make_puzzles(), random.Random(1), []
    for r in p.test + p.bigger + p.dev + p.rl[:300] + p.rl[1000:1300]:
        steps = r["answer"].splitlines(); lhs, rhs = steps[0].split("=")
        out += [(r["answer"], r["nums"]), ("\n".join(steps[:-1]), r["nums"]), ("", r["nums"]),
                ("\n".join([f"{lhs}={int(rhs) + 1}"] + steps[1:]), r["nums"]),
                (r["answer"].replace("*", " × ").replace("/", "÷").replace("=", " = "), r["nums"]),
                (r["answer"] + "\n" + steps[-1], r["nums"])]
    for _ in range(3000):
        nums = [rng.randint(1, 20) for _ in range(rng.choice((3, 4)))]
        pool, lines = list(nums), []
        for _ in range(rng.randint(1, 4)):
            a, b = rng.choice(pool + [rng.randint(1, 30)]), rng.choice(pool + [rng.randint(1, 30)])
            op = rng.choice("+-*/x×÷")
            c = apply(a, op, b)
            c = c if c is not None and rng.random() < 0.8 else rng.randint(-5, 60)
            lines.append(f"{a}{rng.choice(['', ' '])}{op}{rng.choice(['', ' '])}{b}={c}")
            pool.append(c)
        out.append((rng.choice(["\n", "; ", " then "]).join(lines), nums))
    return out + [(None, [1, 2, 3]), ("I am not sure", [1, 2, 3])]


CASES = texts_to_check()


def test_reference_check_accepts_every_worked_solution():
    p = make_puzzles()
    for r in p.train + p.rl + p.dev + p.test + p.bigger + p.seen + p.rl_seen:
        assert reference.check(r["answer"], r["nums"]) == (True, r["target"])


@pytest.mark.parametrize("text,nums", [
    ("3*18=55\n14+55=69", [14, 18, 3]),          # false arithmetic
    ("3*3=9\n9+14=23", [14, 18, 3]),             # a number used twice
    ("3*18=54", [14, 18, 3]),                    # a number left out
    ("", [14, 18, 3]), (None, [14, 18, 3]),      # no steps
    ("3-18=-15\n14+15=29", [14, 18, 3]),         # a negative intermediate
    ("18/4=4\n4+3=7", [18, 4, 3]),               # division that isn't whole
    ("5*18=90\n90-14=76", [14, 18, 3]),          # a number that wasn't given
])
def test_reference_check_rejects(text, nums):
    assert reference.check(text, nums) == (False, None)


def test_reference_check_matches_validated_and_typed(validated_data):
    validated, notebook = validated_data["check"], typed("check", "def check")
    for text, nums in CASES:
        want = validated(text, nums)
        assert reference.check(text, nums) == want, (text, nums)
        assert notebook(text, nums) == want, (text, nums)


def test_reward_matches_validated_and_typed(validated_data):
    ns = {"check": validated_data["check"]}
    exec(definitions(validated_cell("def reward_value"), ["reward_value", "reward_countdown"]), ns)
    notebook = typed("reward", "def reward", check=typed("check", "def check"))
    p = make_puzzles()
    for text, nums in CASES:
        for completion in (text or "", [{"role": "assistant", "content": text or ""}]):
            for target in (p.test[0]["target"], 24, 100):
                want = ns["reward_countdown"]([completion], [nums], [target])
                assert reference.reward([completion], [nums], [target]) == want
                assert notebook(completions=[completion], nums=[nums], target=[target], prompts=None) == want


def test_reward_ladder_values_and_printout(capsys):
    r0, answers = ladder(make_puzzles().rl)
    values = [reference.reward([t], [r0["nums"]], [r0["target"]])[0] for _, t in answers]
    assert [f"{v:.3f}" for v in values] == ["1.000", "0.450", "0.174", "0.000", "0.000", "0.000"]
    reward_ladder(typed("reward", "def reward", check=reference.check), make_puzzles().rl)
    assert capsys.readouterr().out.splitlines() == [       # the validated run's output, line for line
        "puzzle: Numbers: 12 1 4 8 | Target: 22",
        "  its worked solution    -> reward 1.000   12-1=11 ; 8*11=88 ; 88/4=22",
        "  legal, misses by 1     -> reward 0.450   12-1=11 ; 4+8=12 ; 11+12=23",
        "  legal, misses by 10    -> reward 0.174   12-1=11 ; 11-8=3 ; 4*3=12",
        "  false arithmetic       -> reward 0.000   12-1=12 ; 8*11=88 ; 88/4=22",
        "  a number left out      -> reward 0.000   12-1=11 ; 8*11=88",
        "  no steps               -> reward 0.000   I am not sure"]


def test_reward_ladder_flags_a_wrong_reward(capsys):
    reward_ladder(lambda completions, nums, target, **kw: [1.0] * len(completions), make_puzzles().rl)
    out = capsys.readouterr().out
    assert "the reference reward gives 0.450" in out and "from z2v.reference import reward" in out


def test_encode_matches_validated_and_typed(fake_tokenizer, validated_data):
    ns = {"tok": FakeTokenizer(), "SYS": validated_data["SYS"]}
    exec(definitions(validated_cell("def encode"), ["msgs", "render", "ids", "encode"]), ns)
    notebook = typed("encode", "def encode")
    for r in make_puzzles().train[:300]:
        want = ns["encode"](r)
        assert reference.encode(r) == want and notebook(r) == want
    e = reference.encode(make_puzzles().train[1])
    start = e["labels"].count(-100)
    assert FakeTokenizer().decode(e["input_ids"][start:]) == "3*18=54\n14+54=68<|im_end|>\n"
    assert e["labels"][start:] == e["input_ids"][start:]


def test_show_mask_marks_the_boundary(fake_tokenizer, capsys):
    from z2v.explain import show_mask
    show_mask(reference.encode(make_puzzles().train[1]))
    out = capsys.readouterr().out
    assert "<- the answer starts" in out and "warning" not in out


def test_system_prompt_is_the_validated_one(validated_data):
    assert z2v.chat.SYS == validated_data["SYS"]


def test_encode_with_the_real_tokenizer(validated_data):
    "Optional: runs only if transformers is installed and the pinned Qwen tokenizer is already in the local HF cache."
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(z2v.chat.MODEL_ID, revision=z2v.chat.REVISIONS[z2v.chat.MODEL_ID],
                                                         local_files_only=True)
    except Exception:
        pytest.skip("pinned Qwen tokenizer not in the local cache")
    tok.padding_side = "left"
    ns = {"tok": tok, "SYS": validated_data["SYS"]}
    exec(definitions(validated_cell("def encode"), ["msgs", "render", "ids", "encode"]), ns)
    z2v.chat._tokenizers[z2v.chat.MODEL_ID] = tok
    notebook = typed("encode", "def encode")
    for r in make_puzzles().train[:200]:
        assert reference.encode(r) == notebook(r) == ns["encode"](r)
    e = reference.encode(make_puzzles().train[1])
    start = e["labels"].count(-100)
    assert tok.decode(e["input_ids"][start:]) == "3*18=54\n14+54=68<|im_end|>\n"
