import os

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        resolved_model = model_name or os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        self.model = SentenceTransformer(resolved_model)

    def generate_embedding(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()
