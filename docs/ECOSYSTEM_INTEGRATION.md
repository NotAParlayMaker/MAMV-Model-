# Ecosystem integration

End-to-end flow: document → MAMV-Model result → MAMV verification request → MAMV trust receipt → MAMV-IR governed workflow. The latter two artifacts are examples of downstream products and are not generated in this repository.

```python
result = model.produce_result(document, question, include_claim_candidates=True,
                              include_evidence_candidates=True)
verification_request = to_mamv_verification_request(result)
workflow_input = to_mamv_ir_workflow_input(result)
```

The MAMV request preserves source and frame IDs and contains no verdict. The MAMV-IR input has `workflow_state: "unassigned"` and `completion_decision: null`; MAMV-IR assigns those under its own policy.
