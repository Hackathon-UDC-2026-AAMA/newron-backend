import math
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer


class ClusterLabelService:
    def __init__(self) -> None:
        self.stop_words = [
            "de",
            "la",
            "el",
            "en",
            "y",
            "a",
            "que",
            "los",
            "las",
            "un",
            "una",
            "por",
            "para",
            "con",
            "del",
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
        ]

    def build_label(self, texts: list[str], top_k: int = 3) -> tuple[str, list[str]]:
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]
        if not cleaned_texts:
            return "Unlabeled", []

        common_terms = self._extract_common_terms(cleaned_texts)

        top_terms: list[str] = []
        try:
            vectorizer = TfidfVectorizer(stop_words=self.stop_words, token_pattern=r"(?u)\b\w\w+\b")
            matrix = vectorizer.fit_transform(cleaned_texts)

            if matrix.shape[1] > 0:
                scores = matrix.mean(axis=0).A1
                terms = vectorizer.get_feature_names_out()
                top_indices = scores.argsort()[-top_k:][::-1]
                top_terms = [terms[index] for index in top_indices if scores[index] > 0]
        except ValueError:
            top_terms = []

        merged_terms = []
        for term in common_terms + top_terms:
            if term and term not in merged_terms:
                merged_terms.append(term)
            if len(merged_terms) >= top_k:
                break

        if not merged_terms:
            return "Unlabeled", []

        primary_label = self._format_label(merged_terms[0])
        keywords = [self._format_label(term) for term in merged_terms]
        return primary_label, keywords

    def _extract_common_terms(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        tokenized_docs: list[set[str]] = []
        for text in texts:
            tokens = {
                token
                for token in re.findall(r"[a-záéíóúñü0-9]{3,}", text.lower())
                if token not in self.stop_words and not token.isdigit()
            }
            if tokens:
                tokenized_docs.append(tokens)

        if not tokenized_docs:
            return []

        doc_count = len(tokenized_docs)
        min_docs = 1 if doc_count == 1 else max(2, math.ceil(doc_count * 0.4))
        document_frequency = Counter()
        for doc_tokens in tokenized_docs:
            document_frequency.update(doc_tokens)

        ranked = sorted(
            ((term, frequency) for term, frequency in document_frequency.items() if frequency >= min_docs),
            key=lambda item: (-item[1], item[0]),
        )
        return [term for term, _ in ranked[:5]]

    def _format_label(self, value: str) -> str:
        return value.replace("_", " ").strip().title()
