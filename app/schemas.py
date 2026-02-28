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
    normalized_text: str
    metadata: dict[str, Any] = Field(validation_alias="metadata_json")
    embedding: list[float]
    cluster_id: int
    similarity_score: float | None
    created_at: datetime


class ClusterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    cluster_label: str | None
    size: int
    centroid: list[float]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str


class PurgeResponse(BaseModel):
    deleted_items: int
    deleted_clusters: int
