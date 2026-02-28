from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.classifier import classify_input, extract_first_link_url, extract_first_youtube_url
from app.cluster_label_service import ClusterLabelService
from app.clustering_service import ClusteringService
from app.database import Base, engine, get_db
from app.embedding_service import EmbeddingService
from app.models import Cluster, ContentItem
from app.normalizer import normalize_content
from app.schemas import (
    ClusterResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MoveItemClusterRequest,
    MoveItemClusterResponse,
    ItemResponse,
    PurgeResponse,
)

app = FastAPI(title="Semantic Ingestion Backend", version="1.0.0")

embedding_service = EmbeddingService()
clustering_service = ClusteringService()
cluster_label_service = ClusterLabelService()


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE clusters ADD COLUMN IF NOT EXISTS cluster_label VARCHAR(255)"))


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, db: Session = Depends(get_db)) -> IngestResponse:
    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    content_type = classify_input(payload.input)
    canonical_url = _extract_canonical_url(payload.input, content_type)
    if canonical_url is not None:
        existing_item = (
            db.query(ContentItem)
            .filter(
                ContentItem.type == content_type,
                ContentItem.metadata_json["url"].astext == canonical_url,
            )
            .order_by(ContentItem.id.asc())
            .first()
        )
        if existing_item is not None:
            return IngestResponse(
                id=existing_item.id,
                type=existing_item.type,
                cluster_id=existing_item.cluster_id,
                similarity_score=existing_item.similarity_score,
            )

    normalized_text, metadata = normalize_content(payload.input, content_type)
    embedding = embedding_service.generate_embedding(normalized_text)

    cluster, similarity, is_new_cluster = clustering_service.assign_cluster(db, embedding, content_type)

    item = ContentItem(
        original_input=payload.input,
        type=content_type,
        normalized_text=normalized_text,
        metadata_json=metadata,
        embedding=embedding,
        cluster_id=cluster.id,
        similarity_score=similarity,
    )
    db.add(item)
    db.flush()

    if is_new_cluster and not cluster.cluster_label:
        cluster_texts = [row[0] for row in db.query(ContentItem.normalized_text).filter(ContentItem.cluster_id == cluster.id).all()]
        cluster.cluster_label = cluster_label_service.build_label(cluster_texts)

    db.commit()
    db.refresh(item)

    return IngestResponse(
        id=item.id,
        type=item.type,
        cluster_id=item.cluster_id,
        similarity_score=item.similarity_score,
    )


@app.get("/items", response_model=list[ItemResponse])
def get_items(db: Session = Depends(get_db)) -> list[ContentItem]:
    return db.query(ContentItem).order_by(ContentItem.id.asc()).all()


@app.get("/clusters", response_model=list[ClusterResponse])
def get_clusters(db: Session = Depends(get_db)) -> list[Cluster]:
    return db.query(Cluster).order_by(Cluster.id.asc()).all()


@app.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    db.execute(text("SELECT 1"))
    return HealthResponse(status="ok")


@app.post("/admin/purge", response_model=PurgeResponse)
def purge_data(db: Session = Depends(get_db)) -> PurgeResponse:
    deleted_items = db.query(ContentItem).delete(synchronize_session=False)
    deleted_clusters = db.query(Cluster).delete(synchronize_session=False)
    db.commit()
    return PurgeResponse(deleted_items=deleted_items, deleted_clusters=deleted_clusters)


@app.post("/items/{item_id}/move-cluster", response_model=MoveItemClusterResponse)
def move_item_cluster(item_id: int, payload: MoveItemClusterRequest, db: Session = Depends(get_db)) -> MoveItemClusterResponse:
    item = db.query(ContentItem).filter(ContentItem.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    target_cluster = db.query(Cluster).filter(Cluster.id == payload.target_cluster_id).first()
    if target_cluster is None:
        raise HTTPException(status_code=404, detail="Target cluster not found")

    old_cluster_id = item.cluster_id
    if old_cluster_id == target_cluster.id:
        old_cluster_deleted, old_cluster_size = _recompute_cluster_state(db, old_cluster_id)
        db.commit()
        return MoveItemClusterResponse(
            item_id=item.id,
            old_cluster_id=old_cluster_id,
            new_cluster_id=target_cluster.id,
            old_cluster_deleted=old_cluster_deleted,
            old_cluster_size=old_cluster_size,
            new_cluster_size=old_cluster_size or 0,
        )

    item.cluster_id = target_cluster.id
    db.flush()

    old_cluster_deleted, old_cluster_size = _recompute_cluster_state(db, old_cluster_id)
    _, new_cluster_size = _recompute_cluster_state(db, target_cluster.id)

    db.commit()
    db.refresh(item)

    return MoveItemClusterResponse(
        item_id=item.id,
        old_cluster_id=old_cluster_id,
        new_cluster_id=target_cluster.id,
        old_cluster_deleted=old_cluster_deleted,
        old_cluster_size=old_cluster_size,
        new_cluster_size=new_cluster_size or 0,
    )


def _extract_canonical_url(raw_input: str, content_type: str) -> str | None:
    if content_type == "youtube":
        return extract_first_youtube_url(raw_input)
    if content_type == "link":
        return extract_first_link_url(raw_input)
    return None


def _recompute_cluster_state(db: Session, cluster_id: int) -> tuple[bool, int | None]:
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if cluster is None:
        return True, None

    cluster_items = db.query(ContentItem).filter(ContentItem.cluster_id == cluster_id).all()
    if not cluster_items:
        db.delete(cluster)
        db.flush()
        return True, None

    cluster.size = len(cluster_items)
    cluster.centroid = _average_embeddings([item.embedding for item in cluster_items])
    cluster.cluster_label = cluster_label_service.build_label([item.normalized_text for item in cluster_items])
    db.flush()
    return False, cluster.size


def _average_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []

    vector_size = len(embeddings[0])
    sums = [0.0] * vector_size
    for vector in embeddings:
        for index, value in enumerate(vector):
            sums[index] += value

    count = float(len(embeddings))
    return [value / count for value in sums]
