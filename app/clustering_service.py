import math
import os

from sqlalchemy.orm import Session

from app.models import Cluster


class ClusteringService:
    def __init__(self, threshold: float | None = None) -> None:
        env_threshold = os.getenv("SIMILARITY_THRESHOLD")
        self.threshold = threshold if threshold is not None else float(env_threshold or 0.75)
        self.text_threshold = float(os.getenv("TEXT_SIMILARITY_THRESHOLD", os.getenv("SIMILARITY_THRESHOLD_TEXT", 0.45)))
        self.youtube_threshold = float(os.getenv("YOUTUBE_SIMILARITY_THRESHOLD", os.getenv("SIMILARITY_THRESHOLD_YOUTUBE", 0.62)))
        self.link_threshold = float(os.getenv("LINK_SIMILARITY_THRESHOLD", os.getenv("SIMILARITY_THRESHOLD_LINK", 0.55)))

    def assign_cluster(self, db: Session, embedding: list[float], content_type: str | None = None) -> tuple[Cluster, float, bool]:
        clusters = db.query(Cluster).all()
        if not clusters:
            new_cluster = Cluster(centroid=embedding, size=1)
            db.add(new_cluster)
            db.flush()
            return new_cluster, 0.0, True

        if content_type in {"text", "audio"}:
            threshold_to_use = self.text_threshold
        elif content_type == "youtube":
            threshold_to_use = self.youtube_threshold
        elif content_type == "link":
            threshold_to_use = self.link_threshold
        else:
            threshold_to_use = self.threshold

        best_cluster = None
        best_similarity = -1.0

        for cluster in clusters:
            similarity = cosine_similarity(embedding, cluster.centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is not None and best_similarity > threshold_to_use:
            best_cluster.centroid = update_centroid(best_cluster.centroid, best_cluster.size, embedding)
            best_cluster.size += 1
            db.flush()
            return best_cluster, best_similarity, False

        new_cluster = Cluster(centroid=embedding, size=1)
        db.add(new_cluster)
        db.flush()
        return new_cluster, best_similarity, True


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def update_centroid(old_centroid: list[float], old_size: int, new_embedding: list[float]) -> list[float]:
    if old_size <= 0:
        return new_embedding
    weighted_sum = [old_value * old_size + new_value for old_value, new_value in zip(old_centroid, new_embedding)]
    return [value / (old_size + 1) for value in weighted_sum]
