"""Zero to Verifiable: the helper package behind the workshop notebook. A good reading order:
puzzles.py, chat.py, reference.py, model.py, measure.py, train.py, explain.py, compare.py.

`from z2v import *` gives the notebook everything below. Names that need torch, transformers, peft or trl load
on first use, so the pure-Python parts (and the tests) work on a laptop without the GPU stack.
"""
import importlib

from .puzzles import STEP, apply, floor, make_puzzles, show_puzzles

_WHERE = {
    "setup": ".model", "load_model": ".model",
    "render": ".chat", "ids": ".chat",
    "evaluate": ".measure",
    "train_sft": ".train", "train_grpo": ".train",
    "show_mask": ".explain", "explain_sft": ".explain", "reward_ladder": ".explain", "explain_grpo": ".explain",
    "room_for_rl": ".compare", "results": ".compare", "operator_mix": ".compare",
    "LoraConfig": "peft", "TrainingArguments": "transformers", "GRPOConfig": "trl",     # the real library classes
}

__all__ = ["STEP", "apply", "make_puzzles", "show_puzzles", "floor", *_WHERE]


def __getattr__(name):
    "Import a name from its module the first time it is used."
    if name not in _WHERE:
        raise AttributeError(f"module 'z2v' has no attribute {name!r}")
    value = getattr(importlib.import_module(_WHERE[name], __name__), name)
    globals()[name] = value
    return value
