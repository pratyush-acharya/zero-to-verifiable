"""The chat format: the system prompt, the tokenizer, and render(), the one place a puzzle becomes text.

transformers is imported on first use, so this module (and the tests) load without the GPU stack.
"""
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISIONS = {MODEL_ID: "7ae557604adf67be50417f59c2c2f167def9a775"}   # pinned: an upstream template change can't change the task

SYS = ("You solve Countdown puzzles: combine all the given numbers, each exactly once, using + - * / to reach the "
       "target. Every intermediate result must be a positive whole number. Write one step per line, like 4*5=20, "
       "and stop when you reach the target.")

current = {"model_id": MODEL_ID}      # load_model() switches this if you load a different model
_tokenizers = {}


def msgs(p):
    "The conversation for one puzzle: a system turn and a user turn."
    return [{"role": "system", "content": SYS}, {"role": "user", "content": p["prompt"]}]


def tokenizer():
    "The current model's tokenizer, loaded once, with left padding for batched generation."
    model_id = current["model_id"]
    if model_id not in _tokenizers:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, revision=REVISIONS.get(model_id))
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"        # generation needs left padding; SFT batches are padded by train.collate()
        _tokenizers[model_id] = tok
    return _tokenizers[model_id]


def render(p, answer=None):
    "A puzzle as chat text. With an answer: a full training example. Without: the assistant turn is left open."
    m = msgs(p) + ([{"role": "assistant", "content": answer}] if answer is not None else [])
    return tokenizer().apply_chat_template(m, tokenize=False, add_generation_prompt=answer is None)


def ids(text):
    "The token ids of a piece of text (the chat template already holds the special tokens)."
    return tokenizer()(text, add_special_tokens=False)["input_ids"]
