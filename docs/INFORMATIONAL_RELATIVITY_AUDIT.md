# Informational Relativity audit (pre-implementation)

## Existing coverage

`MAMVModel.produce_result()` was the only public path that emitted an `InferenceFrame`. `answer()`, `answer_file()`, `EducationSession`, and `ConversationSession.ask()` returned `Answer`/education values without frame metadata. Direct, CoT, self-consistency, self-refine, integrated and fragmented paths therefore had no consistently portable provenance.

## Gaps

* The prior frame held only backend class name, source IDs, strategy, two strategy booleans, and a context hash. It omitted artifact revisions, tokenizer/adapter identity, retrieval settings/scores, chunk/source locations, generation settings, grounding scope, session history, timestamps, assumptions, limitations, and typed context-loss warnings.
* `produce_result()` used its own SHA-256 seed while result identity reused that same seed; neither represented excluded chunk contents, model revisions, adapter/tokenizer identity, nor separated creation time from frame identity.
* `answer_file()` silently substituted local retrieval after an empty external retrieval; session truncation and integration budget loss were not represented in a frame. Fragment disagreement and empty retrieval were only text/trace behavior.
* Confidence fields conflated stated answer confidence and grounding heuristic confidence; validation was absent. Coherence was separately recorded but had no explicit protection from being treated as grounding.
* Frame construction was duplicated conceptually: inference paths constructed context independently and only `produce_result()` constructed a partial frame. Adapters propagated only IDs/source IDs and omitted warnings, artifacts, and limitations attached to the frame.

This report was completed before the implementation work below.
