"""Looking inside: the label mask, the per-token SFT loss, the reward ladder, and one GRPO group worked by hand.

torch is imported inside the two functions that run the model, so the rest loads without the GPU stack.
"""
import math

import numpy as np

from .chat import msgs, tokenizer
from .puzzles import derivations

_before = {}     # explain_sft remembers the last probabilities it showed per puzzle, to show them rising


def show_mask(example, context=6):
    "Where the loss looks: tokens labelled -100 are ignored; the answer tokens keep their ids as labels."
    tok, x, y = tokenizer(), example["input_ids"], example["labels"]
    assert len(x) == len(y) == len(example["attention_mask"]), "input_ids, labels and attention_mask must be the same length"
    start = next((i for i, l in enumerate(y) if l != -100), len(y))
    print(f"{len(x)} tokens: {start} with label -100 (no loss), then {len(x) - start} the loss is computed on\n")
    print("ignored by the loss:\n  " + tok.decode(x[:start]).replace("\n", "\n  "))
    print(f"\n{'position':>8}  {'token':<12} {'input_id':>8}  {'label':>6}")
    for i in range(max(0, start - context), len(x)):
        print(f"{i:>8}  {tok.decode([x[i]])!r:<12} {x[i]:>8}  {y[i]:>6}" + ("" if i != start else "   <- the answer starts"))
    if any(y[i] not in (-100, x[i]) for i in range(len(x))) or -100 in y[start:]:
        print("\nwarning: after the prompt, every label should equal its input_id (check your encode)")


def explain_sft(model, row, encode):
    "The model's probability for each answer token, and the SFT loss for this example: the sum of −log p."
    import torch
    from .model import dtype
    tok, e = tokenizer(), encode(row)
    was_training = model.training; model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype()):
        logits = model(input_ids=torch.tensor([e["input_ids"]], device=model.device)).logits[0].float()
    if was_training: model.train()
    logp = torch.log_softmax(logits[:-1], -1)              # position t predicts token t + 1
    key, rows = row["prompt"] + row["answer"], []
    for t, label in enumerate(e["labels"][1:]):
        if label != -100: rows.append((tok.decode([label]), logp[t, label].item()))
    before = _before.get(key)
    print(row["prompt"].replace("\n", " | "), " ->  the worked solution, token by token\n")
    print(f"  {'token':<12} {'p':>6} {'−log p':>8}" + ("   p last time" if before else ""))
    for i, (piece, lp) in enumerate(rows):
        print(f"  {piece!r:<12} {math.exp(lp):6.3f} {-lp:8.3f}" + (f"   {math.exp(before[i]):6.3f}" if before else ""))
    total = -sum(lp for _, lp in rows)
    print(f"\n  sum of −log p = {total:.2f}   <- the SFT loss for this example"
          f"\n  (train/loss in TensorBoard is the mean per answer token over a batch; here {total / len(rows):.3f})")
    _before[key] = [lp for _, lp in rows]


def ladder(rows):
    "Six answers to the first 4-number puzzle: its worked solution, two legal misses, and three illegal answers."
    r0 = next(r for r in rows if r["n"] == 4)
    misses = [d for d in derivations(r0["nums"]) if d[0] != r0["target"]]

    def miss_by(off):
        "A legal path that misses the target by about `off`."
        v, steps = min(misses, key=lambda d: abs(abs(d[0] - r0["target"]) - off))
        return f"legal, misses by {abs(v - r0['target'])}", "\n".join(steps)

    first = r0["answer"].splitlines(); lhs, rhs = first[0].split("=")
    return r0, [("its worked solution", r0["answer"]), miss_by(1), miss_by(10),
                ("false arithmetic", "\n".join([f"{lhs}={int(rhs) + 1}"] + first[1:])),
                ("a number left out", "\n".join(first[:-1])), ("no steps", "I am not sure")]


def reward_ladder(reward, rows):
    "Score six answers to one puzzle with your reward, called the way GRPO calls it: keyword columns, chat messages."
    from .reference import reward as reference
    r0, answers = ladder(rows)
    print("puzzle:", r0["prompt"].replace("\n", " | "))
    differs = False
    for label, text in answers:
        call = dict(prompts=[msgs(r0)], completions=[[{"role": "assistant", "content": text}]],
                    nums=[r0["nums"]], target=[r0["target"]])
        got, want = reward(**call)[0], reference(**call)[0]
        note = "" if math.isclose(got, want) else f"   <- the reference reward gives {want:.3f}"
        differs |= bool(note)
        print(f"  {label:<22} -> reward {got:.3f}   {text.replace(chr(10), ' ; ')[:60]}{note}")
    if differs:
        print("\nyour reward differs from the reference; to train with the reference:  from z2v.reference import reward")


def explain_grpo(model, row, reward, cfg):
    "One GRPO group by hand: sample answers to one puzzle, score them, and turn the scores into advantages."
    from .measure import generate
    G = cfg.num_generations
    texts = generate(model, [row], k=G, max_new_tokens=cfg.max_completion_length, temperature=cfg.temperature)[0]
    r = np.array(reward(prompts=[msgs(row)] * G, completions=[[{"role": "assistant", "content": t}] for t in texts],
                        nums=[row["nums"]] * G, target=[row["target"]] * G), float)
    mean, std = r.mean(), r.std(ddof=1)                    # TRL's group std uses Bessel's correction
    scaled = cfg.scale_rewards != "none"
    adv = (r - mean) / (std + 1e-4) if scaled else r - mean
    print(f"{row['prompt'].replace(chr(10), ' | ')}   {G} samples at temperature {cfg.temperature}\n")
    print(f"  {'reward':>6}  {'advantage':>9}   answer")
    for t, ri, ai in sorted(zip(texts, r, adv), key=lambda z: -z[1]):
        print(f"  {ri:6.3f}  {ai:+9.2f}   {t.replace(chr(10), ' ; ')[:70]}")
    how = "(reward − mean) / std" if scaled else "reward − mean"
    print(f"\n  group mean {mean:.3f}, std {std:.3f}  ->  advantage = {how}")
    print("  positive advantage: that answer's tokens are made more likely; negative: less likely.")
    if np.isclose(std, 0):
        print("  all answers scored the same, so every advantage is 0: this puzzle teaches nothing this step.")
