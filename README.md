# Zero to Verifiable

**A 60-minute undergraduate workshop: teach a small language model to solve Countdown puzzles, first by imitation (SFT), then by trial and error against a checker (GRPO), on a free Colab GPU.**

Countdown: combine all the given numbers, each exactly once, with `+ − × ÷` to hit a target (`14 18 3 → 68`: `3*18=54`, `14+54=68`). Finding a solution is a search; checking one takes a dozen lines of code. SFT shows the model one solution per puzzle; GRPO rewards whichever correct solution it finds.

Students type along in Colab. They type only what is being taught: the checker, the label mask, the reward, and the real `LoraConfig`, `TrainingArguments` and `GRPOConfig` objects. Everything else lives in the small `z2v` helper package in this repository.

## Open in Colab

| notebook | for |
|---|---|
| [`notebooks/workshop_typealong.ipynb`](https://colab.research.google.com/github/pratyush-acharya/zero-to-verifiable/blob/main/notebooks/workshop_typealong.ipynb) | students: the same notebook with empty code cells, each saying what to type and roughly how many lines |
| [`notebooks/workshop.ipynb`](https://colab.research.google.com/github/pratyush-acharya/zero-to-verifiable/blob/main/notebooks/workshop.ipynb) | the instructor, and any student who falls behind: complete and runnable top to bottom |

## How to run it

1. Open a notebook in Colab, then `Runtime ▸ Change runtime type ▸ T4 GPU`.
2. Run the setup cell. It installs this repository (`pip install git+…`), which pins `transformers==5.22.1`, `trl==1.18.1` and `peft==0.21.3`; Colab's own `torch` is left alone. `setup()` checks the GPU, seeds everything, downloads the model weights (~1 GB) and opens TensorBoard.
3. Work through the ten stops. About 20 GPU-minutes in total: ~4 min SFT, ~12 min GRPO, ~3 min evaluation. The two long cells are marked ⏱: start them, then switch to the slides.

| stop | what you type | lines |
|---|---|---|
| 1 · Setup | the install line, `from z2v import *`, `setup()` | 3 |
| 2 · Puzzles and the checker | `make_puzzles()`, `show_puzzles()`, **`check(text, nums)`**, `floor()` | 15 |
| 3 · Text to tokens: the mask | **`encode(p)`** with the `-100` label mask, `show_mask()` | 6 |
| 4 · The model and LoRA | **`LoraConfig(...)`**, `load_model(...)` | 7 |
| 5 · Before training | `evaluate(..., adapter=False)`, `explain_sft()` | 2 |
| 6 · SFT ⏱ | **`TrainingArguments(...)`**, `train_sft(..., minutes=4)`, `explain_sft()` | 9 |
| 7 · Room for RL | `evaluate()`, `room_for_rl()` | 2 |
| 8 · The reward | **`reward(completions, nums, target, **kw)`**, `reward_ladder()` | 10 |
| 9 · GRPO ⏱ | **`GRPOConfig(...)`**, `explain_grpo()`, `train_grpo(..., minutes=12)` | 11 |
| 10 · Results | `evaluate()`, `results()`, `operator_mix()` | 3 |

Line counts leave out blank lines and comments. `results()` ends with one line of this run's numbers for the slides, for example `base=3.0&sft=36.3&rl=57.0&avg_sft=23.3&avg_rl=56.8&any_sft=69.3&any_rl=63.3&rand=5.2&four_sft=26.0&four_rl=49.3`.

## If something breaks

- **A function you typed doesn't work:** run `from z2v.reference import check, encode, reward` and carry on. Those are the reference versions, and they replace yours from that cell on. `evaluate()` refuses a `check` that disagrees with the rules, and `reward_ladder()` flags a reward that differs from the reference.
- **You fell behind:** open the complete notebook (`notebooks/workshop.ipynb`) and run it from the top.
- **Out of memory:** a T4 has about 15 GB, so the configs already split each update into smaller passes (SFT 16 × 2, GRPO 16 × 4). If it still runs out, halve `per_device_train_batch_size` again and double `gradient_accumulation_steps`: each update stays the same size.

## What's inside `z2v`

Each module is short and meant to be read; start with `puzzles.py`. Names that need `torch` load on first use, so the pure-Python parts import on any laptop.

| module | contents |
|---|---|
| `puzzles.py` | the rules (`apply`, `STEP`), the generator, `make_puzzles(seed=2026)`, the exhaustive solver (`derivations`), `floor`, `show_puzzles` |
| `chat.py` | the system prompt, the tokenizer, `render(p, answer=None)`, `ids(text)` |
| `reference.py` | reference `check`, `encode` and `reward`: the fallback for students |
| `model.py` | `setup()`, `load_model(model_id, lora)` |
| `measure.py` | `evaluate(...)`: greedy answers plus 8 samples at temperature 1, scored by your `check` |
| `train.py` | `train_sft`, `train_grpo`, the timing probe, the dev-curve and wall-clock callbacks, and what is added to your configs |
| `explain.py` | `show_mask`, `explain_sft`, `reward_ladder`, `explain_grpo` |
| `compare.py` | `room_for_rl`, `results` (table, bootstrap intervals, paired differences, the sentence, the chart, the live numbers), `operator_mix` |

The public API, all available after `from z2v import *`:

- `setup()`: check the GPU, seed everything (3407), cache the pinned weights, open TensorBoard on `/content/runs`.
- `make_puzzles(seed=2026)`: every split (`train`, `rl`, `dev`, `test`, `bigger`, `seen`, `rl_seen`); seed 2026 reproduces the validated run.
- `show_puzzles(puzzles)`: split sizes and a few worked examples.
- `floor(puzzles)`: what the exhaustive solver says about the test puzzles, including how often a random legal path hits.
- `STEP`, `apply(a, op, b)`: the step regex and the rules, for your `check`.
- `render(p, answer=None)`, `ids(text)`: the chat text of a puzzle and its token ids, for your `encode`.
- `show_mask(example)`: the tokens around the prompt/answer boundary with their labels.
- `load_model(model_id, lora)`: the model in 16-bit (bf16 on Ampere or newer, fp16 otherwise) with your LoRA adapter; it prints the trainable parameters.
- `evaluate(model, puzzles, check, adapter=True)`: greedy plus 8 samples per test puzzle; `adapter=False` measures the base weights.
- `explain_sft(model, p, encode)`: each answer token's probability and −log p, and their sum (the SFT loss for that example).
- `train_sft(model, rows, encode, cfg, minutes=4)`: times one step, sets `max_steps` to fit the budget, trains, and stops at the wall clock.
- `room_for_rl(base, sft)`: greedy vs any of 8.
- `reward_ladder(reward, rows)`: six answers to one puzzle scored by your reward, called the way GRPO calls it.
- `explain_grpo(model, p, reward, cfg)`: one group by hand: 8 live samples → rewards → mean and std → advantages.
- `train_grpo(model, rows, reward, cfg, minutes=12)`: GRPO with a dev curve every 25 steps and a wall-clock stop.
- `results(base, sft, rl)`, `operator_mix(sft, rl)`: the final comparison.
- `LoraConfig`, `TrainingArguments`, `GRPOConfig`: the real classes from `peft`, `transformers` and `trl`.

**What the helper adds to your configs.** The knobs you type are kept. `train_sft` adds `output_dir=/content/runs/sft`, TensorBoard logging, bf16 or fp16, `save_strategy="no"`, `remove_unused_columns=False` and `max_steps` from the timing probe (capped at 2 epochs). `train_grpo` adds `output_dir=/content/runs/grpo`, TensorBoard logging, bf16 or fp16, `gradient_checkpointing=False` and `save_strategy="no"`. Knobs you leave at the library default get the validated values: `logging_steps=5` and `seed=3407` for SFT; `scale_rewards="group"`, `max_grad_norm=0.1`, `lr_scheduler_type="constant_with_warmup"`, `warmup_steps=10`, `max_steps=1000` (the wall clock stops it first), `logging_steps=1` and `seed=3407` for GRPO. The configs are copied with `dataclasses.replace`, so the libraries' own checks run again on the result.

## The validation run (L4, 300 new test puzzles)

These numbers come from the single-file notebook this package was refactored from (`tests/fixtures/validated_workshop_60min.ipynb`, an unchanged copy). The refactor reproduces its puzzles, checker, model, settings and evaluation; it has not yet been re-run end to end on a GPU.

| test, 300 puzzles | greedy | 95% CI | 4 numbers | avg of 8 | any of 8 |
|---|---|---|---|---|---|
| a random legal path | 5.2 | | 2.5 | 5.2 | 31.6 |
| base | 3.0 | 1.3 – 5.0 | 0.0 | 1.0 | 7.0 |
| after SFT | 36.3 | 31.0 – 41.7 | 26.0 | 23.3 | 69.3 |
| after GRPO | **57.0** | 51.3 – 62.7 | 49.3 | 56.8 | 63.3 |

Caveats: validated on an L4, not a T4. A T4 is slower, so it gets fewer steps in the same minutes; the time budgets keep the session on schedule. "RL beat more SFT" was not tested.

## Tests

```bash
pip install numpy pandas matplotlib pytest
pytest
```

The tests need no GPU and download nothing. They check that `make_puzzles()` reproduces the validated notebook's puzzles exactly (by running its data cell), the checker and reward against the validated versions and the notebook's typed versions, the solver facts, `results()` on simulated runs (including the live-numbers line), and that both notebooks are valid, in sync, and within the typing budget. With `torch`, `transformers`, `trl` and `peft` installed (CPU is enough), the tests also check that the typed configs plus the helper's additions equal the validated configs field by field, and that `target_modules="all-linear"` adapts exactly the validated seven projections.

## Before the first workshop

- The install line in both notebooks is pinned to a commit (`@<commit sha>`), so every student installs exactly the code that was tested. If you change `z2v/`, push, then move the pin in both notebooks to the new commit.
- Do one full GPU run of `notebooks/workshop.ipynb` and compare it with the table above.

## License

MIT, see [`LICENSE`](LICENSE). The model, `Qwen/Qwen2.5-0.5B-Instruct`, is Apache-2.0 and is downloaded from the Hugging Face Hub at a pinned revision.
