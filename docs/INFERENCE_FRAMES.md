# Inference frames

Every `MAMVModel.answer`, file answer, portable result, education answer, and conversation answer carries an immutable `InferenceFrame`. `build_inference_frame` canonicalizes identity-relevant fields and gives identical inputs identical frame IDs. `created_at` is deliberately excluded from that identity. Frames contain hashed documents/chunks rather than prompts, credentials, or hidden states.

Warnings are typed `FrameWarning` values. Context, retrieval, session, artifact, generation, and grounding fields describe what was available—not truth.
