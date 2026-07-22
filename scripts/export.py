#!/usr/bin/env python3
"""Export a trained checkpoint using Hugging Face safe serialization."""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
AutoModelForCausalLM.from_pretrained(a.model).save_pretrained(a.output, safe_serialization=True)
AutoTokenizer.from_pretrained(a.model).save_pretrained(a.output)
