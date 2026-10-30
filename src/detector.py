"""RAG faithfulness and hallucination evaluation utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd


@dataclass
class ClaimEvaluation:
    claim: str
    entailment: float
    contradiction: float
    neutral: float
    supported: bool
    entity_discrepancy: bool


class RAGFaithfulnessDetector:
    """Evaluate whether an answer is grounded in retrieved RAG context.

    The default NLI model is a compact MNLI model suitable for portfolio demos.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
        support_threshold: float = 0.60,
    ):
        self.model_name = model_name
        self.support_threshold = support_threshold
        self._classifier = None

    def _load_model(self):
        if self._classifier is None:
            from transformers import pipeline

            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
            )
        return self._classifier

    @staticmethod
    def split_claims(answer: str) -> List[str]:
        """Split a short answer into sentence-like claims."""
        claims = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
        return [c.strip() for c in claims if c.strip()]

    @staticmethod
    def _entities(text: str) -> set[str]:
        """Extract simple capitalized entities for discrepancy analysis.

        This intentionally avoids a large NLP model. It is a heuristic and should
        not be treated as a full named-entity recognition system.
        """
        patterns = re.findall(
            r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b|\b\d{4}\b",
            text,
        )
        stop = {"The", "A", "An", "This", "That", "It", "In", "On", "For"}
        return {p for p in patterns if p not in stop}

    def _classify_claim(self, claim: str, context: str) -> Dict[str, float]:
        classifier = self._load_model()
        outputs = classifier({"text": context, "text_pair": claim})

        # transformers versions/model configs can return either a flat list
        # or nested lists.
        labels = outputs[0] if outputs and isinstance(outputs[0], list) else outputs
        scores = {item["label"].lower(): float(item["score"]) for item in labels}

        def find_score(*names: str) -> float:
            for name in names:
                if name in scores:
                    return scores[name]
            return 0.0

        return {
            "entailment": find_score("entailment", "label_2"),
            "contradiction": find_score("contradiction", "label_0"),
            "neutral": find_score("neutral", "label_1"),
        }

    def evaluate(
        self,
        query: str,
        contexts: Sequence[str],
        answer: str,
    ) -> dict:
        """Evaluate an LLM answer against retrieved documents."""
        if not contexts:
            raise ValueError("At least one retrieved context document is required.")
        if not answer.strip():
            raise ValueError("The answer must not be empty.")

        combined_context = "\n\n".join(contexts)
        claims = self.split_claims(answer)
        context_entities = self._entities(combined_context)

        rows = []
        for claim in claims:
            scores = self._classify_claim(claim, combined_context)
            claim_entities = self._entities(claim)
            discrepancy = bool(claim_entities - context_entities)
            supported = scores["entailment"] >= self.support_threshold and not discrepancy

            rows.append(
                ClaimEvaluation(
                    claim=claim,
                    supported=supported,
                    entity_discrepancy=discrepancy,
                    **scores,
                ).__dict__
            )

        report = pd.DataFrame(rows)
        raw_score = float(report["entailment"].mean()) if not report.empty else 0.0
        discrepancy_rate = (
            float(report["entity_discrepancy"].mean()) if not report.empty else 0.0
        )
        faithfulness_score = max(0.0, raw_score - 0.20 * discrepancy_rate)

        return {
            "query": query,
            "answer": answer,
            "faithfulness_score": round(faithfulness_score, 4),
            "supported_claims": int(report["supported"].sum()),
            "total_claims": len(report),
            "report": report,
            "claims": rows,
        }

    @staticmethod
    def to_dataframe(result: dict) -> pd.DataFrame:
        """Return claim-level evaluation rows as a pandas DataFrame."""
        return pd.DataFrame(result["claims"])
