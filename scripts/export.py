#!/usr/bin/env python3
"""Optionally merge a LoRA adapter into standalone Hugging Face weights."""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
p.add_argument(
    "--merge-adapter", action="store_true", help="merge a PEFT adapter into full weights"
)
a = p.parse_args()
if a.merge_adapter:
    from peft import PeftConfig, PeftModel

    config = PeftConfig.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, a.model).merge_and_unload()
else:
    model = AutoModelForCausalLM.from_pretrained(a.model)
model.save_pretrained(a.output, safe_serialization=True)
AutoTokenizer.from_pretrained(a.model).save_pretrained(a.output)
