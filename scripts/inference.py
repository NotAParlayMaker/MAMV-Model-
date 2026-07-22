#!/usr/bin/env python3
import argparse
from pathlib import Path
from mamv_model import MAMVModel
from mamv_model.config import load_config, reasoning_answer_kwargs
from mamv_model.model_result import save_model_result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--document", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--reasoning-config", default="configs/reasoning.yaml")
    p.add_argument("--output-json", type=Path)
    p.add_argument("--include-claim-candidates", action="store_true")
    p.add_argument("--include-evidence-candidates", action="store_true")
    p.add_argument("--include-relation-candidates", action="store_true")
    p.add_argument("--compact", action="store_true")
    a = p.parse_args()
    reasoning = load_config(Path(a.reasoning_config)).reasoning
    model = MAMVModel.load(a.model, require_grounding=reasoning.require_grounding)
    kwargs = reasoning_answer_kwargs(reasoning)
    if a.output_json:
        result = model.produce_result(
            a.document, a.question, include_claim_candidates=a.include_claim_candidates,
            include_evidence_candidates=a.include_evidence_candidates,
            include_relation_candidates=a.include_relation_candidates, **kwargs,
        )
        save_model_result(result, a.output_json, compact=a.compact)
        print(result.answer)
    else:
        print(model.answer(a.document, a.question, **kwargs).text)


if __name__ == "__main__":
    main()
