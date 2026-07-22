# Education use: prototype guardrails and limits

MAMV is a research/prototyping tool, **not** a vetted classroom product. Its
`EducationSession` API is feedback-only: it exposes a reasoning trace, source
locations, and separate model-stated, consensus, and lexical-grounding
confidence fields. It does not provide grades, pass/fail outcomes, or a blended
"correctness" score. Teachers and institutions remain responsible for review
and all consequential decisions.

## Data handling

The library does not persist submitted documents or student text to disk by
default. File ingestion reads only the path supplied by the caller and performs
local extraction in memory. This repository provides **no FERPA, COPPA, or GDPR
compliance**, no student-data retention policy, and no consent flow. A deployer
must establish lawful processing, access controls, retention/deletion rules,
parental consent where required, and any institutional review before use with
student data.

## Supported readings

`ingest_file()` supports `.txt`, `.md`, text-extractable `.pdf`, and `.docx`.
PDF chunk citations include a page number where available. Scanned/image-only
PDFs fail clearly: OCR is intentionally out of scope for this v1 ingestion path.
Do not treat lexical verification or a model trace as proof that an answer is
correct.

## Student-writing feedback

`critique_claim()` takes a student-written claim and returns feedback plus a
conservative lexical support label. It never rewrites the student's submission.
`estimate_genericity("Every student finished the reflection.")` may flag broad
quantifiers for discussion; it is a standalone heuristic, not a grading signal.
