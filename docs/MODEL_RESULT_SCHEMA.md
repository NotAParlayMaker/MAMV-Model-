# MAMV model result schema

`mamv-model-result/v1` is a JSON-serializable immutable artifact. Its `inference_frame` identifies the model, visible and excluded chunks, strategy, grounding requirement, revision state, sample agreement, and a context hash. Candidate records always carry that frame ID.

`reasoning_summary` deliberately excludes model reasoning steps. Confidence and grounding fields are model/heuristic signals, not evidence verification. Claim candidates have `status: "unverified"`; proposed relations have `status: "model_proposed"`; retrieval score is not support.

Use `model_result_to_json`, `model_result_from_json`, `save_model_result`, and `load_model_result` for portable interchange. Unknown schema versions are rejected.
