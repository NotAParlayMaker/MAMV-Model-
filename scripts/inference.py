#!/usr/bin/env python3
import argparse
from pathlib import Path
from mamv_model import MAMVModel
from mamv_model.config import load_config, reasoning_answer_kwargs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--reasoning-config", default="configs/reasoning.yaml")
    a = p.parse_args()
    reasoning = load_config(Path(a.reasoning_config)).reasoning
    model = MAMVModel.load(a.model, require_grounding=reasoning.require_grounding)
    print(
        model.answer(a.document, a.question, **reasoning_answer_kwargs(reasoning)).text
    )


if __name__ == "__main__":
    main()
