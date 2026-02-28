import math
import os
import re
from collections import defaultdict

from sqlalchemy.orm import Session

from app.models import Cluster, ContentItem


class ClusteringService:
    def __init__(self, threshold: float | None = None) -> None:
        env_threshold = os.getenv("SIMILARITY_THRESHOLD")
        self.threshold = threshold if threshold is not None else float(env_threshold or 0.75)
        self.text_threshold = float(os.getenv("TEXT_SIMILARITY_THRESHOLD", os.getenv("SIMILARITY_THRESHOLD_TEXT", 0.45)))
        self.text_threshold_cold_start = float(
            os.getenv("TEXT_SIMILARITY_THRESHOLD_COLD_START", os.getenv("SIMILARITY_THRESHOLD_TEXT_COLD_START", 0.52))
        )
        self.file_threshold = float(os.getenv("FILE_SIMILARITY_THRESHOLD", os.getenv("SIMILARITY_THRESHOLD_FILE", 0.72)))
        self.youtube_threshold = float(os.getenv("YOUTUBE_SIMILARITY_THRESHOLD", os.getenv("SIMILARITY_THRESHOLD_YOUTUBE", 0.62)))
        self.link_threshold = float(os.getenv("LINK_SIMILARITY_THRESHOLD", os.getenv("SIMILARITY_THRESHOLD_LINK", 0.55)))
        self.text_theme_guard_enabled = os.getenv("TEXT_THEME_GUARD_ENABLED", "true").strip().lower() == "true"
        self.text_theme_min_overlap = int(os.getenv("TEXT_THEME_MIN_OVERLAP", "1"))
        self.text_theme_high_similarity_override = float(os.getenv("TEXT_THEME_HIGH_SIMILARITY_OVERRIDE", "0.78"))
        self.text_theme_min_cluster_size = int(os.getenv("TEXT_THEME_MIN_CLUSTER_SIZE", "3"))
        self.adaptive_min_cluster_size = 3
        self.adaptive_sigma_factor = 1.0
        self.adaptive_global_min_threshold = 0.65
        self.adaptive_recalc_every = 5
        self._adaptive_threshold_cache: dict[int, tuple[int, float]] = {}

    def assign_cluster(
        self,
        db: Session,
        embedding: list[float],
        content_type: str | None = None,
        source_text: str | None = None,
        source_keywords: list[str] | None = None,
    ) -> tuple[Cluster, float, bool]:
        clusters = db.query(Cluster).all()
        if not clusters:
            new_cluster = Cluster(centroid=embedding, size=1)
            db.add(new_cluster)
            db.flush()
            return new_cluster, 0.0, True

        if content_type in {"text", "audio"}:
            threshold_to_use = self.text_threshold
        elif content_type == "file":
            threshold_to_use = self.file_threshold
        elif content_type == "youtube":
            threshold_to_use = self.youtube_threshold
        elif content_type == "link":
            threshold_to_use = self.link_threshold
        else:
            threshold_to_use = self.threshold

        best_cluster = None
        best_similarity = -1.0
        best_global_similarity = -1.0
        cluster_item_embeddings = self._load_cluster_item_embeddings(db, clusters)

        for cluster in clusters:
            similarity = cosine_similarity(embedding, cluster.centroid)
            if similarity > best_global_similarity:
                best_global_similarity = similarity

            if not self._passes_thematic_guard(
                cluster,
                content_type,
                source_text,
                source_keywords,
                similarity,
            ):
                continue

            if similarity > best_similarity:
                dynamic_threshold = self._resolve_threshold_for_cluster(
                    cluster=cluster,
                    threshold_to_use=threshold_to_use,
                    content_type=content_type,
                    cluster_item_embeddings=cluster_item_embeddings,
                )
                if similarity > dynamic_threshold:
                    best_similarity = similarity
                    best_cluster = cluster

        if best_cluster is not None:
            best_cluster.centroid = update_centroid(best_cluster.centroid, best_cluster.size, embedding)
            best_cluster.size += 1
            db.flush()
            return best_cluster, best_similarity, False

        new_cluster = Cluster(centroid=embedding, size=1)
        db.add(new_cluster)
        db.flush()
        return new_cluster, best_global_similarity, True

    def _passes_thematic_guard(
        self,
        cluster: Cluster,
        content_type: str | None,
        source_text: str | None,
        source_keywords: list[str] | None,
        similarity: float,
    ) -> bool:
        if content_type not in {"text", "audio"}:
            return True
        if not self.text_theme_guard_enabled:
            return True
        if cluster.size < self.text_theme_min_cluster_size:
            return True
        if similarity >= self.text_theme_high_similarity_override:
            return True

        incoming_tokens = self._build_theme_tokens(source_text, source_keywords)
        cluster_tokens = self._normalize_tokens(cluster.cluster_keywords or [])

        if not incoming_tokens or not cluster_tokens:
            return True

        overlap = len(incoming_tokens.intersection(cluster_tokens))
        return overlap >= self.text_theme_min_overlap

    def _build_theme_tokens(self, source_text: str | None, source_keywords: list[str] | None) -> set[str]:
        tokens = self._normalize_tokens(source_keywords or [])
        if source_text:
            text_tokens = re.findall(r"[a-záéíóúñü]{4,}", source_text.lower())
            tokens.update(text_tokens)
        return tokens

    def _normalize_tokens(self, values: list[str]) -> set[str]:
        normalized: set[str] = set()
        for value in values:
            for token in re.findall(r"[a-záéíóúñü]{4,}", str(value).lower()):
                normalized.add(token)
        return normalized

    def _load_cluster_item_embeddings(self, db: Session, clusters: list[Cluster]) -> dict[int, list[list[float]]]:
        if not clusters:
            return {}

        cluster_ids = [cluster.id for cluster in clusters]
        rows = db.query(ContentItem.cluster_id, ContentItem.embedding).filter(ContentItem.cluster_id.in_(cluster_ids)).all()

        grouped: dict[int, list[list[float]]] = defaultdict(list)
        for cluster_id, item_embedding in rows:
            if item_embedding:
                grouped[int(cluster_id)].append(item_embedding)
        return grouped

    def _resolve_threshold_for_cluster(
        self,
        cluster: Cluster,
        threshold_to_use: float,
        content_type: str | None,
        cluster_item_embeddings: dict[int, list[list[float]]],
    ) -> float:
        static_threshold = threshold_to_use
        if content_type in {"text", "audio"} and cluster.size <= 1:
            static_threshold = self.text_threshold_cold_start

        if cluster.size < self.adaptive_min_cluster_size:
            return static_threshold

        cached = self._adaptive_threshold_cache.get(cluster.id)
        if cached is not None:
            cached_size, cached_threshold = cached
            if cluster.size % self.adaptive_recalc_every != 0 and cached_size <= cluster.size:
                return max(self.adaptive_global_min_threshold, cached_threshold)

        item_embeddings = cluster_item_embeddings.get(cluster.id, [])
        if len(item_embeddings) < self.adaptive_min_cluster_size:
            return static_threshold

        similarities = [cosine_similarity(item_embedding, cluster.centroid) for item_embedding in item_embeddings]
        if not similarities:
            return static_threshold

        dynamic_threshold = self._compute_dynamic_threshold(similarities)
        bounded_threshold = max(self.adaptive_global_min_threshold, dynamic_threshold)
        self._adaptive_threshold_cache[cluster.id] = (cluster.size, bounded_threshold)
        return bounded_threshold

    def _compute_dynamic_threshold(self, similarities: list[float]) -> float:
        mu = sum(similarities) / len(similarities)
        variance = sum((value - mu) ** 2 for value in similarities) / len(similarities)
        sigma = math.sqrt(max(variance, 0.0))
        return mu - (self.adaptive_sigma_factor * sigma)


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
