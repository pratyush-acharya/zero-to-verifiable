"""Setup and the model: GPU check, seeds, cached weights, TensorBoard, and the base model with a LoRA adapter."""
import os
import random

import numpy as np
import torch

from .chat import MODEL_ID, REVISIONS, current, tokenizer

RUNS, OUT, SEED = "/content/runs", "/content/ckpt", 3407      # everything under RUNS shows up in TensorBoard


def bf16():
    "True on Ampere or newer (compute capability 8+). A T4 only emulates bf16 (slowly), so it gets fp16."
    return torch.cuda.get_device_capability(0)[0] >= 8


def dtype():
    "The 16-bit type for weights and autocast: bf16 where the GPU has it, fp16 otherwise."
    return torch.bfloat16 if bf16() else torch.float16


def setup(model_id=MODEL_ID):
    "Check the GPU, seed everything, download the model weights now, and open TensorBoard."
    import peft, transformers, trl
    from huggingface_hub import snapshot_download
    assert torch.cuda.is_available(), "No GPU -> Runtime > Change runtime type > T4 GPU"
    cap, vram = torch.cuda.get_device_capability(0), torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  {vram:.1f} GiB  bf16={bf16()}")
    print({"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__})

    os.makedirs(RUNS, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    snapshot_download(model_id, revision=REVISIONS.get(model_id))     # ~1 GB, fetched now so load_model() reads the cache
    print("model files cached")

    try:                                       # TensorBoard stays empty until training logs its first step
        from IPython import get_ipython
        get_ipython().run_line_magic("load_ext", "tensorboard")
        get_ipython().run_line_magic("tensorboard", f"--logdir {RUNS}")
    except Exception as e:                     # not in a notebook
        print(f"TensorBoard not opened ({type(e).__name__}); run: tensorboard --logdir {RUNS}")


def load_model(model_id, lora):
    "The base model in 16-bit on the GPU, wrapped with a fresh LoRA adapter built from `lora`."
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    current["model_id"] = model_id
    tokenizer()                                # render() and generation use the matching tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=REVISIONS.get(model_id), dtype=dtype(),
                                                 device_map={"": 0})
    model.config.use_cache = False             # training; generate() turns the KV cache back on
    lora.task_type = lora.task_type or "CAUSAL_LM"     # PEFT's causal-LM wrapper (as in the validated run)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    print("VRAM after load: %.2f GiB" % (torch.cuda.max_memory_allocated() / 2**30))
    return model
