from src.detector import RAGFaithfulnessDetector


def test_split_claims():
    claims = RAGFaithfulnessDetector.split_claims(
        "Paris is in France. The city is the capital."
    )
    assert len(claims) == 2


def test_empty_context():
    detector = RAGFaithfulnessDetector()
    try:
        detector.evaluate("q", [], "answer")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_entities():
    entities = RAGFaithfulnessDetector._entities("Paris was founded in 1889.")
    assert "Paris" in entities
    assert "1889" in entities
