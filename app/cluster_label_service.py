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

    def build_label(self, texts: list[str], top_k: int = 3) -> str:
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]
        if not cleaned_texts:
            return "Unlabeled"

        vectorizer = TfidfVectorizer(stop_words=self.stop_words, token_pattern=r"(?u)\b\w\w+\b")
        matrix = vectorizer.fit_transform(cleaned_texts)

        if matrix.shape[1] == 0:
            return "Unlabeled"

        scores = matrix.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        top_indices = scores.argsort()[-top_k:][::-1]
        top_terms = [terms[index] for index in top_indices if scores[index] > 0]

        if not top_terms:
            return "Unlabeled"
        return " / ".join(top_terms)
