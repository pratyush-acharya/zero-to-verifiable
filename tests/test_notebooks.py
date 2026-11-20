"""Both notebooks: valid, compilable, in sync, within the typing budget, and using the knobs the slides show."""
import ast
import builtins
import json
import re

import pytest
from conftest import NOTEBOOKS, cells

import z2v

INSTALL_RE = re.compile(r'!pip install -q "git\+https://github\.com/pratyush-acharya/zero-to-verifiable@(main|[0-9a-f]{40})"')
INSTALL = next(line for line in cells(NOTEBOOKS["workshop"], "code")[0].splitlines() if line.startswith("!pip"))
STOPS = ["Setup", "Puzzles and the checker", "Text to tokens: the mask", "The model and LoRA", "Before training",
         "SFT", "Room for RL", "The reward", "GRPO", "Results"]


def python(src):
    "A code cell without its shell (!) and magic (%) lines."
    return "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%")))


def stops(nb):
    "Stop number -> the code typed in that stop, from the '## N · Title' headings."
    out, current = {}, None
    for c in json.loads(NOTEBOOKS[nb].read_text(encoding="utf-8"))["cells"]:
        src = "".join(c["source"])
        m = re.match(r"## (\d+) · ", src)
        if c["cell_type"] == "markdown" and m:
            current = int(m.group(1)); out[current] = []
        elif c["cell_type"] == "code" and current:
            out[current].append(src)
    return out


def typed_lines(src):
    "Lines a student has to type: not blank, not only a comment."
    return [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_valid_json_and_every_code_cell_compiles(nb):
    data = json.loads(NOTEBOOKS[nb].read_text(encoding="utf-8"))
    assert data["nbformat"] == 4 and data["metadata"]["accelerator"] == "GPU"
    for i, c in enumerate(data["cells"]):
        assert c["cell_type"] in ("markdown", "code") and isinstance(c["source"], list)
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            compile(python("".join(c["source"])), f"{nb} cell {i}", "exec")


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_install_line_appears_once_and_is_pinned_to_this_repo(nb):
    text = NOTEBOOKS[nb].read_text(encoding="utf-8")
    assert INSTALL_RE.fullmatch(INSTALL), INSTALL
    assert text.count("github.com/pratyush-acharya/zero-to-verifiable@") == 1
    assert INSTALL in cells(NOTEBOOKS[nb], "code")[0].splitlines()


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_stop_headings_are_the_contract(nb):
    headings = [re.match(r"## (\d+) · (.+)", c.splitlines()[0]) for c in cells(NOTEBOOKS[nb], "markdown")]
    assert [(int(m.group(1)), m.group(2)) for m in headings if m] == list(enumerate(STOPS, 1))


def test_notebooks_share_markdown_and_layout():
    a, b = (json.loads(p.read_text(encoding="utf-8"))["cells"] for p in NOTEBOOKS.values())
    assert [c["cell_type"] for c in a] == [c["cell_type"] for c in b]
    assert [c["source"] for c in a if c["cell_type"] == "markdown"] == [c["source"] for c in b if c["cell_type"] == "markdown"]


def test_typealong_cells_hold_only_instructions():
    for i, src in enumerate(cells(NOTEBOOKS["typealong"], "code")):
        lines = typed_lines(src)
        assert lines == ([INSTALL] if i == 0 else []), src
        assert "# Type:" in src and re.search(r"\(~?\d+ lines?[,)]", src), src


def test_typing_budget_per_stop():
    counts = {n: sum(len(typed_lines(s)) for s in srcs) for n, srcs in stops("workshop").items()}
    assert set(counts) == set(range(1, 11))
    assert max(counts.values()) <= 15, counts
    print("typed lines per stop:", counts)


def test_notebook_only_uses_exported_names():
    tree = ast.parse("\n".join(python(c) for c in cells(NOTEBOOKS["workshop"], "code")))
    defined = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    defined |= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    defined |= {a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments) for a in n.args + [n.kwarg] if a}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = used - defined - set(dir(builtins)) - set(z2v.__all__)
    assert not missing, missing


def calls(name):
    "Keyword arguments of every call to `name` in the instructor notebook."
    tree = ast.parse("\n".join(python(c) for c in cells(NOTEBOOKS["workshop"], "code")))
    return [({k.arg: ast.literal_eval(k.value) for k in n.keywords if k.arg},
             [ast.unparse(a) for a in n.args])
            for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]


def test_configs_have_the_validated_knobs():
    assert calls("LoraConfig") == [({"r": 16, "lora_alpha": 16, "target_modules": "all-linear", "lora_dropout": 0.0}, [])]
    assert calls("TrainingArguments") == [({"learning_rate": 2e-4, "per_device_train_batch_size": 16,
                                            "gradient_accumulation_steps": 2,
                                            "lr_scheduler_type": "linear", "warmup_steps": 20}, [])]
    assert calls("GRPOConfig") == [({"num_generations": 8, "temperature": 1.0, "learning_rate": 2e-5, "beta": 0.0,
                                     "max_completion_length": 64, "per_device_train_batch_size": 16,
                                     "gradient_accumulation_steps": 4}, [])]
    assert calls("load_model") == [({}, ["'Qwen/Qwen2.5-0.5B-Instruct'", "lora"])]
    assert calls("train_sft") == [({"minutes": 4}, ["model", "puzzles.train", "encode", "sft_cfg"])]
    assert calls("train_grpo") == [({"minutes": 12}, ["model", "puzzles.rl", "reward", "grpo_cfg"])]
    assert [kw for kw, _ in calls("evaluate")] == [{"adapter": False}, {}, {}]
