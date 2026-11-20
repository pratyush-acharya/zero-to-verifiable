"""Shared helpers: the validated notebook (an unchanged copy in fixtures/), the workshop notebooks, and a fake tokenizer."""
import ast
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATED = ROOT / "tests" / "fixtures" / "validated_workshop_60min.ipynb"
NOTEBOOKS = {"workshop": ROOT / "notebooks" / "workshop.ipynb",
             "typealong": ROOT / "notebooks" / "workshop_typealong.ipynb"}


def cells(path, kind=None):
    "Cell sources of a notebook, optionally only one cell type."
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if kind is None or c["cell_type"] == kind]


def definitions(src, names):
    "Only the named top-level defs/assignments of a cell, as source (so imports the cell needs on a GPU are skipped)."
    keep = []
    for node in ast.parse(src).body:
        defined = ({node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                   {t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)})
        if defined & set(names):
            keep.append(ast.get_source_segment(src, node))
    return "\n\n".join(keep)


def validated_cell(marker):
    "The validated notebook's code cell containing `marker`."
    return next(c for c in cells(VALIDATED, "code") if marker in c)


@pytest.fixture(scope="session")
def validated_data():
    "Run the validated notebook's data cell (pure Python) and return its namespace."
    ns = {"random": random, "re": re, "np": np}
    exec(compile(validated_cell("SPLITS = {"), "validated data cell", "exec"), ns)
    return ns


def notebook_cell(marker, nb="workshop"):
    "The instructor notebook's code cell containing `marker`."
    return next(c for c in cells(NOTEBOOKS[nb], "code") if marker in c)


class FakeTokenizer:
    "Qwen's chat layout with one id per character: enough to test the label mask without downloading anything."
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}

    def decode(self, ids):
        return "".join(map(chr, ids))


@pytest.fixture
def fake_tokenizer(monkeypatch):
    "Make render(), ids() and show_mask() use the fake tokenizer."
    import z2v.chat, z2v.explain
    tok = FakeTokenizer()
    monkeypatch.setattr(z2v.chat, "tokenizer", lambda: tok)
    monkeypatch.setattr(z2v.explain, "tokenizer", lambda: tok)
    return tok
