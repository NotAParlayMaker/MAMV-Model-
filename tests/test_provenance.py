from mamv_model.provenance import (
    compare_evaluation_reports,
    dataset_manifest,
    unavailable,
)


def test_dataset_fingerprint_changes_with_preprocessing_and_revision():
    a = dataset_manifest(
        upstream_dataset="x",
        upstream_revision="1",
        split="train",
        preprocessing_configuration={"v": 1},
        output_hashes={},
        row_counts={},
        generated_timestamp="fixed",
    )
    b = dataset_manifest(
        upstream_dataset="x",
        upstream_revision="1",
        split="train",
        preprocessing_configuration={"v": 2},
        output_hashes={},
        row_counts={},
        generated_timestamp="fixed",
    )
    c = dataset_manifest(
        upstream_dataset="x",
        upstream_revision="2",
        split="train",
        preprocessing_configuration={"v": 1},
        output_hashes={},
        row_counts={},
        generated_timestamp="fixed",
    )
    assert a["manifest_id"] != b["manifest_id"] and a["manifest_id"] != c["manifest_id"]


def test_unavailable_and_incompatible_reports():
    assert unavailable()["status"] == "not_evaluated"
    a = {
        "frame": {"dataset_manifest_id": "a", "split_fingerprint": "x"},
        "summary_metrics": {"exact_match": 1},
    }
    b = {
        "frame": {"dataset_manifest_id": "b", "split_fingerprint": "x"},
        "summary_metrics": {"exact_match": 0},
    }
    assert compare_evaluation_reports(a, b)["classification"] == "not_comparable"
