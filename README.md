# RAG Faithfulness & Hallucination Detector

A lightweight evaluation toolkit for Retrieval-Augmented Generation (RAG) systems. It accepts a user query, retrieved context documents, and an LLM answer, then evaluates whether answer claims are supported by the retrieved evidence.

## Features

- Splits an LLM answer into simple claims.
- Uses Natural Language Inference (NLI) to estimate whether each claim is entailed by retrieved context.
- Detects entity discrepancies using lightweight named-entity extraction.
- Produces claim-level and answer-level faithfulness scores.
- Generates a pandas evaluation report.
- Includes a Jupyter notebook with reproducible examples.
- Includes unit tests.

## Project structure

```text
rag-faithfulness-hallucination-detector/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── detector.py
├── tests/
│   └── test_detector.py
└── notebooks/
    └── rag_hallucination_eval.ipynb
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first NLI evaluation downloads the configured Hugging Face model.

## Quick start

```python
from src.detector import RAGFaithfulnessDetector

detector = RAGFaithfulnessDetector()

result = detector.evaluate(
    query="When was the Eiffel Tower completed?",
    contexts=[
        "The Eiffel Tower was completed in 1889 for the Exposition Universelle."
    ],
    answer="The Eiffel Tower was completed in 1889."
)

print(result["faithfulness_score"])
print(result["claims"])
```

## Interpretation

The claim-level score is based on the NLI model's entailment probability. The final score is the mean entailment probability across answer claims, with an additional penalty when extracted entities are absent from the retrieved context.

This is an evaluation aid, not a perfect hallucination oracle. NLI models can make mistakes, and claim splitting/entity extraction are intentionally lightweight.

## Tests

```bash
pytest -q
```

## Notebook

Open:

```text
notebooks/rag_hallucination_eval.ipynb
```

The notebook demonstrates claim evaluation and pandas reporting.

## License

MIT
