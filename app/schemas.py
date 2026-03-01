from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IngestRequest(BaseModel):
    input: str | list[str] = Field(
        ...,
        description="Texto único o lista de textos/URLs para ingesta.",
        examples=[
            "Nota técnica sobre embeddings y clustering",
            ["https://example.com", "Resumen de reunión de producto"],
        ],
    )


class IngestResponse(BaseModel):
    id: int = Field(..., description="ID del item guardado.")
    type: str = Field(..., description="Tipo detectado: text, link, youtube, audio o file.")
    cluster_id: int = Field(..., description="Cluster asignado al item.")
    similarity_score: float | None = Field(None, description="Similitud contra el centroide del cluster elegido.")


class ProcessingResponse(BaseModel):
    status: str = Field(..., description="Estado de encolado del procesamiento.")
    mode: str = Field(..., description="Modo de ingesta encolado: single, bulk, audio o file.")
    queued: int = Field(..., description="Cantidad de tareas encoladas por la petición.")


class BulkIngestRequest(BaseModel):
    inputs: list[str] = Field(..., description="Lista de entradas a procesar en lote.")
    continue_on_error: bool = Field(True, description="Si es false, se detiene el lote al primer error.")


class BulkIngestItemResult(BaseModel):
    index: int = Field(..., description="Posición del elemento en el lote de entrada.")
    input: str = Field(..., description="Valor original enviado para este elemento.")
    success: bool = Field(..., description="Indica si el elemento se procesó correctamente.")
    result: IngestResponse | None = Field(None, description="Resultado de ingesta cuando success=true.")
    error: str | None = Field(None, description="Error de procesamiento cuando success=false.")


class BulkIngestResponse(BaseModel):
    total: int = Field(..., description="Cantidad total recibida en el lote.")
    processed: int = Field(..., description="Cantidad realmente procesada.")
    succeeded: int = Field(..., description="Cantidad de elementos procesados con éxito.")
    failed: int = Field(..., description="Cantidad de elementos con error.")
    results: list[BulkIngestItemResult] = Field(..., description="Resultado detallado por elemento.")


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


class ItemContentResponse(BaseModel):
    item_id: int
    type: str
    title: str | None = None
    content: str
    download_url: str | None = None
