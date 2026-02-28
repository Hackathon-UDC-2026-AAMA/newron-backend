from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IngestRequest(BaseModel):
    input: str


class IngestResponse(BaseModel):
    id: int
    type: str
    cluster_id: int
    similarity_score: float | None


class ItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_input: str
    type: str
    normalized_text: str = Field(validation_alias="original_input")
    processed_text: str = Field(validation_alias="normalized_text")
    metadata: dict[str, Any] = Field(validation_alias="metadata_json")
    embedding: list[float]
    cluster_id: int
    similarity_score: float | None
    created_at: datetime


class ClusterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    cluster_label: str | None
    cluster_description: str | None = None
    cluster_keywords: list[str] = Field(default_factory=list)
    size: int
    centroid: list[float]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str


class PurgeResponse(BaseModel):
    deleted_items: int
    deleted_clusters: int


class MoveItemClusterRequest(BaseModel):
    target_cluster_id: int


class MoveItemClusterResponse(BaseModel):
    item_id: int
    old_cluster_id: int
    new_cluster_id: int
    old_cluster_deleted: bool
    old_cluster_size: int | None
    new_cluster_size: int
