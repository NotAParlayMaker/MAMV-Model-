"""Run lexical genericity/quantifier benchmark without a checkpoint."""

import json
from pathlib import Path
from mamv_model import estimate_genericity

rows = [json.loads(line) for line in Path("evals/genericity.jsonl").read_text().splitlines()]
results = [estimate_genericity(row["text"]) for row in rows]
print(
    json.dumps(
        {
            "genericity_accuracy": sum(
                x.is_generic == r["genericity_label"] for x, r in zip(results, rows)
            )
            / len(rows),
            "quantifier_accuracy": sum(
                (x.quantifier or "none") == r["quantifier"] for x, r in zip(results, rows)
            )
            / len(rows),
        },
        indent=2,
    )
)
