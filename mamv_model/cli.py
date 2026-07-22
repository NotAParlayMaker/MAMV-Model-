"""Command-line interface for MAMV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import DocumentQA


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a grounded question about a local document.")
    parser.add_argument("command", nargs="?", choices=["ask"], default="ask")
    parser.add_argument("--doc", "--document", dest="document", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    qa = DocumentQA()
    try:
        qa.add_file(str(args.document))
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    result = qa.ask(args.question)
    payload = {"answer": result.answer, "confidence": result.confidence, "citations": [citation.__dict__ for citation in result.citations]}
    if args.json:
        print(json.dumps(payload, indent=2))
    elif result.answer:
        print(result.answer)
        for citation in result.citations:
            location = ", ".join(item for item in [citation.source_file,
                                 f"page {citation.page_number}" if citation.page_number else None,
                                 citation.section_title] if item)
            print(f"\n[{location or citation.document_id}; score={citation.score}]\n{citation.passage}")
    else:
        print("No sufficiently grounded answer found.")


if __name__ == "__main__":
    main()
