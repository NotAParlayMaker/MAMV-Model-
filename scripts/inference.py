#!/usr/bin/env python3
import argparse
from pathlib import Path
from mamv_model import MAMVModel
from mamv_model.config import load_config, reasoning_answer_kwargs
from mamv_model.model_result import save_model_result, model_result_to_json, load_model_result, compare_inference_frames


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
    p.add_argument("--show-frame", action="store_true")
    p.add_argument("--save-frame", type=Path)
    p.add_argument("--compare-frame", type=Path)
    p.add_argument("--show-frame-transition", action="store_true")
    p.add_argument("--strict-reproducibility", action="store_true")
    a = p.parse_args()
    reasoning = load_config(Path(a.reasoning_config)).reasoning
    model = MAMVModel.load(a.model, require_grounding=reasoning.require_grounding)
    kwargs = reasoning_answer_kwargs(reasoning)
    if a.output_json or a.show_frame or a.save_frame or a.compare_frame or a.show_frame_transition or a.strict_reproducibility:
        result = model.produce_result(
            a.document, a.question, include_claim_candidates=a.include_claim_candidates,
            include_evidence_candidates=a.include_evidence_candidates,
            include_relation_candidates=a.include_relation_candidates, **kwargs,
        )
        if a.output_json: save_model_result(result, a.output_json, compact=a.compact)
        if a.save_frame: a.save_frame.write_text(model_result_to_json(result, compact=a.compact))
        if a.strict_reproducibility:
            codes = {w.code for w in result.inference_frame.warnings}
            if {"MODEL_REVISION_UNPINNED", "TOKENIZER_REVISION_UNPINNED", "ADAPTER_REVISION_UNPINNED"} & codes or (kwargs.get("temperature", 0) and "seed" not in kwargs): raise SystemExit("strict reproducibility failed: unpinned artifact or stochastic seed")
        print(result.answer)
        if a.show_frame: print(model_result_to_json(result, compact=False))
        if a.compare_frame: print(compare_inference_frames(load_model_result(a.compare_frame).inference_frame, result.inference_frame))
        if a.show_frame_transition and result.frame_transition: print(result.frame_transition)
    else:
        print(model.answer(a.document, a.question, **kwargs).text)


if __name__ == "__main__":
    main()
