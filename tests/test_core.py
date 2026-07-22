from mamv_model.genericity import estimate_genericity
from mamv_model.metrics import exact_match, f1
from mamv_model.retrieval import InMemoryRetriever


def test_metrics():
    assert exact_match("The Answer", "answer") == 1 and f1("red blue", "blue") > 0


def test_genericity():
    assert estimate_genericity("All birds fly.").quantifier == "all"


def test_retrieval():
    assert InMemoryRetriever({"x": "red apple"}).retrieve("apple")[0].source_id == "x"
