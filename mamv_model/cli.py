"""Command-line interface for MAMV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import DocumentQA


def _read_document(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise SystemExit("PDF support requires: pip install 'mamv-model[pdf]'") from error
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a grounded question about a local document.")
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    qa = DocumentQA()
    qa.add_document(args.document.name, _read_document(args.document))
    result = qa.ask(args.question)
    payload = {"answer": result.answer, "confidence": result.confidence, "citations": [citation.__dict__ for citation in result.citations]}
    if args.json:
        print(json.dumps(payload, indent=2))
    elif result.answer:
        print(result.answer)
        for citation in result.citations:
            print(f"\n[{citation.document_id}#{citation.passage_id}; score={citation.score}]\n{citation.passage}")
    else:
        print("No sufficiently grounded answer found.")


if __name__ == "__main__":
    main()
