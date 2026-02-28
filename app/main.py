from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
import secrets
from datetime import datetime, timedelta
from io import BytesIO
import qrcode
from fastapi.responses import StreamingResponse
import qrcode_terminal
import json
import re
import os

from app.classifier import classify_input, extract_all_link_urls, extract_first_link_url, extract_first_youtube_url
from app.cluster_label_service import ClusterLabelService
from app.clustering_service import ClusteringService
from app.database import Base, engine, get_db
from app.embedding_service import EmbeddingService
from app.file_storage_service import ensure_storage_dir, save_uploaded_file
from app.file_text_service import (
    build_file_embedding_seed,
    build_semantic_views_for_file,
    extract_text_from_file,
    extract_title_from_file,
    SUPPORTED_FILE_EXTENSIONS,
)
from app.llm_category_service import LlmCategoryService
from app.models import Cluster, ContentItem
from app.normalizer import normalize_content
from app.semantic_focus_service import SemanticFocusService
from app.stt_service import transcribe_audio
from app.schemas import (
    BulkIngestItemResult,
    BulkIngestRequest,
    BulkIngestResponse,
    ClusterResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ItemContentResponse,
    MoveItemClusterRequest,
    MoveItemClusterResponse,
    ItemResponse,
    PurgeResponse,
)

OPENAPI_TAGS = [
    {"name": "Auth", "description": "Pairing QR y sesión móvil."},
    {"name": "Ingest", "description": "Entrada de contenido (texto, lista, audio y archivos)."},
    {"name": "Items", "description": "Consulta y recuperación de items indexados."},
    {"name": "Clusters", "description": "Consulta de clusters y agrupaciones."},
    {"name": "Admin", "description": "Operaciones administrativas de mantenimiento."},
    {"name": "Health", "description": "Estado operativo del servicio."},
]

app = FastAPI(
    title="Semantic Ingestion Backend",
    version="1.0.0",
    description=(
        "API de ingesta semántica para frontend. "
        "Permite indexar texto, enlaces, YouTube, audio y archivos; "
        "consultar clusters/items y recuperar archivos categorizados."
    ),
    openapi_tags=OPENAPI_TAGS,
)

embedding_service = EmbeddingService()
clustering_service = ClusteringService()
cluster_label_service = ClusterLabelService()
semantic_focus_service = SemanticFocusService()
llm_category_service = LlmCategoryService()

QR_TOKEN = None
QR_EXPIRATION = None
PAIRED = False
AUDIO_FILE_EXTENSIONS = {".m4a", ".mp3", ".wav"}
TEXT_FILE_FANOUT_EXTENSIONS = {".txt", ".md", ".markdown"}
FILE_LINK_FANOUT_ENABLED = True
FILE_LINK_FANOUT_MIN_LINKS = 2

def get_server_ip():
    # Intenta leer la IP que le pasamos desde afuera
    # Si no existe, usa localhost por defecto
    return os.getenv("HOST_IP", "127.0.0.1")


@app.get("/qr", tags=["Auth"], summary="Generar QR de pairing", description="Devuelve un PNG con el QR para vincular cliente móvil.")
def get_qr():
    ip = get_server_ip()

    payload = {
        "ip": ip,
        "token": QR_TOKEN
    }

    qr_data = json.dumps(payload)

    img = qrcode.make(qr_data)

    print(ip)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

@app.post("/pair", tags=["Auth"], summary="Confirmar pairing", description="Valida el token escaneado desde el QR y marca la sesión como vinculada.")
def pair(secret: str):
    global PAIRED

    if secret != QR_TOKEN:
        raise HTTPException(403, "Invalid")

    if PAIRED:
        return {"status": "already_paired"}

    PAIRED = True
    return {"status": "paired"}

@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_storage_dir()
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE clusters ADD COLUMN IF NOT EXISTS cluster_label VARCHAR(255)"))
        connection.execute(text("ALTER TABLE clusters ADD COLUMN IF NOT EXISTS cluster_description VARCHAR(400)"))
        connection.execute(text("ALTER TABLE clusters ADD COLUMN IF NOT EXISTS cluster_keywords JSONB"))
        connection.execute(text("UPDATE clusters SET cluster_keywords = '[]'::jsonb WHERE cluster_keywords IS NULL"))

    global QR_TOKEN, QR_EXPIRATION

    QR_TOKEN = secrets.token_urlsafe(32)
    QR_EXPIRATION = datetime.utcnow() + timedelta(minutes=5)

    print("\n========== QR LOGIN ==========")
    print("Open http://localhost:8000/qr")
    print("==============================\n")

    ip = get_server_ip()

    payload = {
        "ip": ip,
        "token": QR_TOKEN
    }

    print(payload)

    qr_data = json.dumps(payload)

    print("\n=== PAIR YOUR MOBILE ===\n")
    qrcode_terminal.draw(qr_data)
    print("\nScan this QR with your Expo app\n")


@app.post(
    "/ingest",
    response_model=IngestResponse | BulkIngestResponse,
    tags=["Ingest"],
    summary="Ingesta unitaria o en lote",
    description="Si input es string procesa un item; si input es lista procesa lote y devuelve resultados por elemento.",
)
def ingest(payload: IngestRequest, db: Session = Depends(get_db)) -> IngestResponse | BulkIngestResponse:
    if isinstance(payload.input, list):
        return _ingest_bulk(
            BulkIngestRequest(inputs=payload.input, continue_on_error=True),
            db,
        )
    return _ingest_input(payload.input, db)


def _ingest_bulk(payload: BulkIngestRequest, db: Session) -> BulkIngestResponse:
    max_bulk_items = int(os.getenv("BULK_INGEST_MAX_ITEMS", "100"))
    if not payload.inputs:
        raise HTTPException(status_code=400, detail="inputs must not be empty")
    if len(payload.inputs) > max_bulk_items:
        raise HTTPException(status_code=400, detail=f"Bulk size exceeds limit ({max_bulk_items})")

    results: list[BulkIngestItemResult] = []
    succeeded = 0
    failed = 0
    processed = 0

    for index, input_value in enumerate(payload.inputs):
        try:
            ingest_result = _ingest_input(input_value, db)
            results.append(
                BulkIngestItemResult(
                    index=index,
                    input=input_value,
                    success=True,
                    result=ingest_result,
                )
            )
            succeeded += 1
            processed += 1
        except HTTPException as exc:
            failed += 1
            processed += 1
            error_message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            results.append(
                BulkIngestItemResult(
                    index=index,
                    input=input_value,
                    success=False,
                    error=error_message,
                )
            )
            if not payload.continue_on_error:
                break

    return BulkIngestResponse(
        total=len(payload.inputs),
        processed=processed,
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@app.post(
    "/ingest-audio",
    tags=["Ingest"],
    summary="Ingesta de audio",
    description="Recibe archivo de audio, transcribe y lo indexa como contenido tipo audio.",
)
async def ingest_audio(
    request: Request,
    file: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    db: Session = Depends(get_db),
) -> dict:
    uploaded_file = file or audio
    if uploaded_file is None:
        form = await request.form()
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No se recibió archivo. Usa multipart/form-data con key 'file' o 'audio'.",
                "content_type": request.headers.get("content-type", ""),
                "received_form_keys": list(form.keys()),
            },
        )

    filename = uploaded_file.filename or "audio.m4a"
    extension = f".{filename.lower().split('.')[-1]}" if "." in filename else ""

    if extension not in {".m4a", ".mp3", ".wav"}:
        raise HTTPException(status_code=400, detail="Formato de audio no soportado")

    audio_bytes = await uploaded_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio vacío")

    transcribed_text = transcribe_audio(audio_bytes, suffix=extension)
    if not transcribed_text:
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del audio")

    result = _ingest_input(transcribed_text, db, content_type_override="audio")

    return {
        "filename": filename,
        "transcription": transcribed_text,
        "result": {
            "id": result.id,
            "type": result.type,
            "cluster_id": result.cluster_id,
            "similarity_score": result.similarity_score,
        },
        "status": "processed",
    }


@app.post(
    "/ingest-file",
    tags=["Ingest"],
    summary="Ingesta de archivo",
    description="Guarda el archivo original, extrae texto y genera embedding determinista sin LangExtract.",
)
async def ingest_file(
    request: Request,
    file: UploadFile | None = File(None),
    document: UploadFile | None = File(None),
    file_type: str | None = Form(None),
    metadata: str | None = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    uploaded_file = file or document
    if uploaded_file is None:
        form = await request.form()
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No se recibió archivo. Usa multipart/form-data con key 'file' o 'document'.",
                "content_type": request.headers.get("content-type", ""),
                "received_form_keys": list(form.keys()),
            },
        )

    filename = uploaded_file.filename or "document"
    filename_extension = f".{filename.lower().split('.')[-1]}" if "." in filename else ""
    declared_extension = _resolve_declared_file_extension(file_type=file_type, metadata_raw=metadata)

    if (file_type or metadata) and not declared_extension:
        raise HTTPException(status_code=400, detail="Tipo de archivo declarado no soportado")

    extension = declared_extension or filename_extension
    if extension not in SUPPORTED_FILE_EXTENSIONS and extension not in AUDIO_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado")

    file_bytes = await uploaded_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    storage_info = save_uploaded_file(file_bytes, filename)

    if extension in AUDIO_FILE_EXTENSIONS:
        transcribed_text = transcribe_audio(file_bytes, suffix=extension)
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del audio")

        metadata_overrides = {
            "file_id": storage_info["file_id"],
            "file_name": storage_info["original_filename"],
            "file_storage_path": storage_info["stored_path"],
            "file_size_bytes": storage_info["file_size_bytes"],
            "declared_file_type": file_type,
            "file_extension": extension,
            "ingest_mode": "audio_file",
            "transcription_chars": len(transcribed_text),
        }

        result = _ingest_input(
            transcribed_text,
            db,
            content_type_override="audio",
            original_input_override=f"[AUDIO_FILE] {filename}",
            metadata_overrides=metadata_overrides,
        )

        return {
            "filename": filename,
            "file_id": storage_info["file_id"],
            "file_extension": extension,
            "ingest_mode": "audio_file",
            "transcription_chars": len(transcribed_text),
            "result": {
                "id": result.id,
                "type": result.type,
                "cluster_id": result.cluster_id,
                "similarity_score": result.similarity_score,
            },
            "status": "processed",
        }

    try:
        extracted_text = extract_text_from_file(file_bytes, extension)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not extracted_text:
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del archivo")

    if FILE_LINK_FANOUT_ENABLED and extension in TEXT_FILE_FANOUT_EXTENSIONS:
        extracted_urls = extract_all_link_urls(extracted_text)
        if len(extracted_urls) >= FILE_LINK_FANOUT_MIN_LINKS:
            bulk_result = _ingest_bulk(BulkIngestRequest(inputs=extracted_urls, continue_on_error=True), db)
            return {
                "filename": filename,
                "file_id": storage_info["file_id"],
                "file_extension": extension,
                "ingest_mode": "url_fanout",
                "detected_urls": len(extracted_urls),
                "results": bulk_result.model_dump(),
                "status": "processed",
            }

    file_title = extract_title_from_file(file_bytes, extension, filename)

    file_semantic_views = build_semantic_views_for_file(extracted_text, file_title)
    file_semantic_focus = semantic_focus_service.compose_focus_text(file_semantic_views)
    file_embedding_seed = build_file_embedding_seed(file_semantic_views, file_title)
    if not file_embedding_seed:
        raise HTTPException(status_code=400, detail="No se pudo preparar representación semántica del archivo")

    metadata_overrides = {
        "file_id": storage_info["file_id"],
        "file_name": storage_info["original_filename"],
        "file_storage_path": storage_info["stored_path"],
        "file_size_bytes": storage_info["file_size_bytes"],
        "file_title": file_title,
        "declared_file_type": file_type,
        "extracted_chars": len(extracted_text),
        "index_chars": len(file_embedding_seed),
        "preview_text": extracted_text[:8000],
        "deterministic_file_envelope": True,
        "file_extension": extension,
    }

    ingest_kwargs = {
        "raw_input": file_embedding_seed,
        "db": db,
        "content_type_override": "file",
        "use_semantic_focus": False,
        "original_input_override": f"[FILE] {file_title}",
        "metadata_overrides": metadata_overrides,
    }
    ingest_kwargs["semantic_focus_views_override"] = file_semantic_views
    ingest_kwargs["semantic_focus_override"] = file_semantic_focus
    ingest_kwargs["semantic_focus_source_override"] = "deterministic_file_envelope"

    result = _ingest_input(
        **ingest_kwargs,
    )

    return {
        "filename": filename,
        "file_title": file_title,
        "file_id": storage_info["file_id"],
        "extracted_chars": len(extracted_text),
        "index_chars": len(file_embedding_seed),
        "result": {
            "id": result.id,
            "type": result.type,
            "cluster_id": result.cluster_id,
            "similarity_score": result.similarity_score,
        },
        "status": "processed",
    }


def _ingest_input(
    raw_input: str,
    db: Session,
    content_type_override: str | None = None,
    use_semantic_focus: bool = True,
    original_input_override: str | None = None,
    metadata_overrides: dict | None = None,
    semantic_focus_views_override: dict | None = None,
    semantic_focus_override: str | None = None,
    semantic_focus_source_override: str | None = None,
) -> IngestResponse:
    raw_input = _strip_nul(raw_input)

    if not raw_input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    if content_type_override is not None:
        content_type = content_type_override
        classification_source = "override"
    else:
        content_type = classify_input(raw_input)
        classification_source = "rule-based"
    canonical_url = _extract_canonical_url(raw_input, content_type)
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
    
    normalized_text, metadata = normalize_content(raw_input, content_type)
    normalized_text = _strip_nul(normalized_text)
    metadata = dict(metadata)
    if metadata_overrides:
        metadata.update(metadata_overrides)
    metadata = _sanitize_metadata(metadata)
    metadata["classification_source"] = classification_source
    if semantic_focus_views_override is not None or semantic_focus_override is not None:
        semantic_focus_views = semantic_focus_views_override or {}
        if semantic_focus_override is not None:
            semantic_focus = semantic_focus_override
        elif semantic_focus_views:
            semantic_focus = semantic_focus_service.compose_focus_text(semantic_focus_views)
        else:
            semantic_focus = normalized_text
        semantic_focus_source = semantic_focus_source_override or "provided_override"
    elif use_semantic_focus:
        try:
            semantic_focus_views, semantic_focus_source = semantic_focus_service.build_focus_views(normalized_text, content_type)
            semantic_focus = semantic_focus_service.compose_focus_text(semantic_focus_views)
        except ValueError as exc:
            semantic_focus_views = semantic_focus_service.build_keyword_fallback_views(normalized_text, content_type)
            semantic_focus = semantic_focus_service.compose_focus_text(semantic_focus_views) if semantic_focus_views else normalized_text
            semantic_focus_source = f"keyword_fallback_on_error:{str(exc)}"
    else:
        semantic_focus_views = {}
        semantic_focus = normalized_text
        semantic_focus_source = "disabled_for_file"

    semantic_focus = _strip_nul(semantic_focus)

    if semantic_focus_source.startswith("keyword_fallback_on_error") or semantic_focus_source.startswith("fallback_on_error"):
        embedding_text = _build_embedding_text_canonical(
            content_type=content_type,
            normalized_text=normalized_text,
            original_input=original_input_override or raw_input,
            metadata=metadata,
            semantic_focus_views=semantic_focus_views,
        )
    else:
        embedding_text = _build_embedding_text(normalized_text, semantic_focus, semantic_focus_views)
    embedding_text = _strip_nul(embedding_text)

    metadata["semantic_focus"] = semantic_focus
    metadata["semantic_focus_views"] = semantic_focus_views
    metadata["embedding_text"] = embedding_text
    metadata["semantic_focus_source"] = semantic_focus_source
    metadata.setdefault("pre_focus_text", normalized_text)

    embedding = embedding_service.generate_embedding(embedding_text)

    semantic_keywords = semantic_focus_views.get("keywords", []) if isinstance(semantic_focus_views, dict) else []
    if not isinstance(semantic_keywords, list):
        semantic_keywords = []

    cluster, similarity, is_new_cluster = clustering_service.assign_cluster(
        db,
        embedding,
        content_type,
        source_text=normalized_text,
        source_keywords=[str(keyword) for keyword in semantic_keywords],
    )

    item = ContentItem(
        original_input=original_input_override or raw_input,
        type=content_type,
        normalized_text=semantic_focus,
        metadata_json=metadata,
        embedding=embedding,
        cluster_id=cluster.id,
        similarity_score=similarity,
    )
    db.add(item)
    db.flush()

    _recompute_cluster_state(db, cluster.id)

    db.commit()
    db.refresh(item)

    return IngestResponse(
        id=item.id,
        type=item.type,
        cluster_id=item.cluster_id,
        similarity_score=item.similarity_score,
    )


@app.get(
    "/items/{item_id}/file",
    tags=["Items"],
    summary="Descargar archivo original",
    description="Devuelve el archivo original asociado a un item de tipo file.",
)
def get_item_file(item_id: int, db: Session = Depends(get_db)) -> FileResponse:
    item = db.query(ContentItem).filter(ContentItem.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    if item.type != "file":
        raise HTTPException(status_code=400, detail="Item is not a file")

    stored_path = item.metadata_json.get("file_storage_path")
    filename = item.metadata_json.get("file_name", "document.pdf")
    if not stored_path:
        raise HTTPException(status_code=404, detail="File path not found in metadata")
    if not Path(stored_path).exists():
        raise HTTPException(status_code=404, detail="Stored file not found")

    extension = Path(filename).suffix.lower()
    media_type = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".js": "text/javascript",
        ".jsx": "text/javascript",
        ".mjs": "text/javascript",
        ".cjs": "text/javascript",
        ".ts": "application/typescript",
        ".tsx": "application/typescript",
        ".py": "text/x-python",
        ".java": "text/x-java-source",
        ".go": "text/x-go",
        ".rs": "text/plain",
        ".php": "application/x-httpd-php",
        ".rb": "text/plain",
        ".sh": "application/x-sh",
        ".css": "text/css",
        ".scss": "text/x-scss",
        ".less": "text/plain",
        ".yml": "application/yaml",
        ".yaml": "application/yaml",
        ".xml": "application/xml",
        ".sql": "application/sql",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".csv": "text/csv",
        ".json": "application/json",
        ".html": "text/html",
        ".htm": "text/html",
        ".rtf": "application/rtf",
    }.get(extension, "application/octet-stream")

    return FileResponse(path=stored_path, filename=filename, media_type=media_type)


@app.get(
    "/items",
    response_model=list[ItemResponse],
    tags=["Items"],
    summary="Listar items",
    description="Lista todos los items indexados ordenados por ID ascendente.",
)
def get_items(db: Session = Depends(get_db)) -> list[ContentItem]:
    return db.query(ContentItem).order_by(ContentItem.id.asc()).all()


@app.get(
    "/clusters/{cluster_id}/items",
    response_model=list[ItemResponse],
    tags=["Clusters"],
    summary="Listar items de un cluster",
    description="Devuelve los items pertenecientes a un cluster específico.",
)
def get_cluster_items(cluster_id: int, db: Session = Depends(get_db)) -> list[ContentItem]:
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    return (
        db.query(ContentItem)
        .filter(ContentItem.cluster_id == cluster_id)
        .order_by(ContentItem.id.asc())
        .all()
    )


@app.get(
    "/items/{item_id}/content",
    response_model=ItemContentResponse,
    tags=["Items"],
    summary="Obtener contenido para UI",
    description="Devuelve texto de presentación para chat/UI; en archivos devuelve preview y URL de descarga.",
)
def get_item_content(item_id: int, db: Session = Depends(get_db)) -> ItemContentResponse:
    item = db.query(ContentItem).filter(ContentItem.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    metadata = item.metadata_json or {}

    if item.type == "file":
        preview_text = str(metadata.get("preview_text") or "").strip()
        if not preview_text:
            stored_path = metadata.get("file_storage_path")
            file_name = str(metadata.get("file_name") or "document.pdf")
            if stored_path and Path(stored_path).exists():
                extension = Path(file_name).suffix.lower()
                try:
                    raw_bytes = Path(stored_path).read_bytes()
                    preview_text = extract_text_from_file(raw_bytes, extension)[:8000]
                except Exception:
                    preview_text = ""

        return ItemContentResponse(
            item_id=item.id,
            type=item.type,
            title=str(metadata.get("file_name") or item.original_input),
            content=preview_text or "No se pudo generar previsualización del archivo.",
            download_url=f"/items/{item.id}/file",
        )

    title = None
    if item.type in {"youtube", "link"}:
        title = str(metadata.get("title") or "").strip() or None
        content = str(metadata.get("summary") or metadata.get("description") or item.original_input).strip()
    else:
        content = item.original_input

    return ItemContentResponse(
        item_id=item.id,
        type=item.type,
        title=title,
        content=content,
        download_url=None,
    )


@app.get(
    "/clusters",
    response_model=list[ClusterResponse],
    tags=["Clusters"],
    summary="Listar clusters",
    description="Devuelve todos los clusters con centroides y metadata de categoría.",
)
def get_clusters(db: Session = Depends(get_db)) -> list[Cluster]:
    return db.query(Cluster).order_by(Cluster.id.asc()).all()


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Verifica conectividad básica con base de datos y estado del servicio.",
)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    db.execute(text("SELECT 1"))
    return HealthResponse(status="ok")


@app.post(
    "/admin/purge",
    response_model=PurgeResponse,
    tags=["Admin"],
    summary="Purgar datos",
    description="Elimina todos los items y clusters. Operación destructiva.",
)
def purge_data(db: Session = Depends(get_db)) -> PurgeResponse:
    deleted_items = db.query(ContentItem).delete(synchronize_session=False)
    deleted_clusters = db.query(Cluster).delete(synchronize_session=False)
    db.commit()
    return PurgeResponse(deleted_items=deleted_items, deleted_clusters=deleted_clusters)

@app.post(
    "/items/{item_id}/move-cluster",
    response_model=MoveItemClusterResponse,
    tags=["Clusters"],
    summary="Mover item entre clusters",
    description="Reasigna manualmente un item a otro cluster y recalcula ambos clusters.",
)
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


def _resolve_declared_file_extension(file_type: str | None, metadata_raw: str | None) -> str | None:
    candidates: list[str] = []
    if file_type and file_type.strip():
        candidates.append(file_type.strip())

    metadata_candidate = _extract_file_type_from_metadata(metadata_raw)
    if metadata_candidate:
        candidates.append(metadata_candidate)

    for candidate in candidates:
        resolved = _normalize_file_type_to_extension(candidate)
        if resolved:
            return resolved
    return None


def _extract_file_type_from_metadata(metadata_raw: str | None) -> str | None:
    if not metadata_raw or not metadata_raw.strip():
        return None

    raw = metadata_raw.strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw

    if isinstance(parsed, str):
        return parsed
    if not isinstance(parsed, dict):
        return None

    for key in ["file_type", "type", "mime_type", "content_type", "extension", "ext"]:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_file_type_to_extension(value: str) -> str | None:
    normalized = value.strip().lower()
    if not normalized:
        return None

    if normalized.startswith("."):
        extension = normalized
    elif "/" in normalized:
        mime_map = {
            "application/pdf": ".pdf",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "text/javascript": ".js",
            "application/javascript": ".js",
            "application/x-javascript": ".js",
            "application/typescript": ".ts",
            "text/typescript": ".ts",
            "text/x-python": ".py",
            "text/x-java-source": ".java",
            "text/x-go": ".go",
            "application/x-sh": ".sh",
            "text/css": ".css",
            "application/yaml": ".yaml",
            "text/yaml": ".yaml",
            "application/xml": ".xml",
            "text/xml": ".xml",
            "application/sql": ".sql",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/csv": ".csv",
            "application/json": ".json",
            "text/html": ".html",
            "application/rtf": ".rtf",
            "text/rtf": ".rtf",
        }
        extension = mime_map.get(normalized)
        if extension is None:
            return None
    else:
        extension = f".{normalized}"

    if extension in SUPPORTED_FILE_EXTENSIONS:
        return extension
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
    cluster_label, cluster_description, cluster_keywords = _build_cluster_label(cluster_items)
    cluster.cluster_label = cluster_label
    cluster.cluster_description = cluster_description
    cluster.cluster_keywords = cluster_keywords
    db.flush()

    merged_clusters = _merge_clusters_by_common_keywords(db, cluster)
    if merged_clusters > 0:
        return _recompute_cluster_state(db, cluster.id)

    return False, cluster.size


def _merge_clusters_by_common_keywords(db: Session, primary_cluster: Cluster) -> int:
    minimum_overlap = max(1, int(os.getenv("CLUSTER_MERGE_MIN_KEYWORD_OVERLAP", "1")))
    primary_tokens = _normalize_cluster_keywords(primary_cluster.cluster_keywords)
    if not primary_tokens:
        return 0

    candidate_clusters = db.query(Cluster).filter(Cluster.id != primary_cluster.id).all()
    merged_count = 0

    for candidate_cluster in candidate_clusters:
        candidate_tokens = _normalize_cluster_keywords(candidate_cluster.cluster_keywords)
        if not candidate_tokens:
            continue

        overlap = primary_tokens.intersection(candidate_tokens)
        if len(overlap) < minimum_overlap:
            continue

        db.query(ContentItem).filter(ContentItem.cluster_id == candidate_cluster.id).update(
            {ContentItem.cluster_id: primary_cluster.id},
            synchronize_session=False,
        )
        db.delete(candidate_cluster)
        merged_count += 1

    if merged_count > 0:
        db.flush()

    return merged_count


def _normalize_cluster_keywords(values: list[str] | None) -> set[str]:
    if not values:
        return set()

    normalized_tokens: set[str] = set()
    for value in values:
        for token in re.findall(r"[a-záéíóúñü0-9]{3,}", str(value).lower()):
            normalized_tokens.add(token)
    return normalized_tokens


def _build_cluster_label(cluster_items: list[ContentItem]) -> tuple[str, str | None, list[str]]:
    cluster_texts = [_extract_item_cluster_content(item) for item in cluster_items]
    cluster_texts = [text for text in cluster_texts if text]
    if not cluster_texts:
        cluster_texts = [item.normalized_text for item in cluster_items if item.normalized_text]

    tfidf_label, tfidf_keywords = cluster_label_service.build_label(cluster_texts)
    llm_label, llm_description = llm_category_service.generate_category(tfidf_keywords, cluster_texts, fallback_label=tfidf_label)
    return llm_label, llm_description, tfidf_keywords


def _extract_item_cluster_content(item: ContentItem) -> str:
    metadata = item.metadata_json if isinstance(item.metadata_json, dict) else {}

    def clean_text(value: object, max_chars: int = 450) -> str:
        cleaned = _strip_nul(str(value or "")).strip()
        if not cleaned:
            return ""
        return cleaned[:max_chars]

    content_type = (item.type or "").strip().lower()

    if content_type in {"text", "audio"}:
        for candidate in [metadata.get("pre_focus_text"), item.original_input, item.normalized_text]:
            resolved = clean_text(candidate)
            if resolved:
                return resolved

    if content_type == "link":
        for candidate in [
            metadata.get("summary"),
            metadata.get("description"),
            metadata.get("title"),
            metadata.get("pre_focus_text"),
            item.original_input,
        ]:
            resolved = clean_text(candidate)
            if resolved:
                return resolved

    if content_type == "youtube":
        title = clean_text(metadata.get("title"), max_chars=180)
        description = clean_text(metadata.get("description"), max_chars=320)
        tags_value = metadata.get("tags", [])
        tags = ", ".join(str(tag).strip() for tag in tags_value if str(tag).strip()) if isinstance(tags_value, list) else ""
        tags = clean_text(tags, max_chars=140)

        joined_parts = [part for part in [title, description, tags] if part]
        if joined_parts:
            return " | ".join(joined_parts)

        for candidate in [metadata.get("pre_focus_text"), item.original_input]:
            resolved = clean_text(candidate)
            if resolved:
                return resolved

    if content_type == "file":
        semantic_views = metadata.get("semantic_focus_views", {})
        summary = ""
        if isinstance(semantic_views, dict):
            summary = clean_text(semantic_views.get("summary"), max_chars=320)
        for candidate in [
            metadata.get("preview_text"),
            summary,
            metadata.get("file_title"),
            metadata.get("pre_focus_text"),
            item.original_input,
        ]:
            resolved = clean_text(candidate)
            if resolved:
                return resolved

    for candidate in [metadata.get("pre_focus_text"), item.original_input, item.normalized_text]:
        resolved = clean_text(candidate)
        if resolved:
            return resolved

    return ""


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


def _build_embedding_text(normalized_text: str, semantic_focus: str, semantic_focus_views: dict) -> str:
    base_text = _strip_nul(normalized_text).strip()
    focus_text = _strip_nul(semantic_focus).strip()

    view_parts: list[str] = []
    topic = _strip_nul(str(semantic_focus_views.get("topic", ""))).strip()
    domain = _strip_nul(str(semantic_focus_views.get("domain", ""))).strip()
    summary = _strip_nul(str(semantic_focus_views.get("summary", ""))).strip()
    intent = _strip_nul(str(semantic_focus_views.get("intent", ""))).strip()
    expanded_context = _strip_nul(str(semantic_focus_views.get("expanded_context", ""))).strip()

    if topic:
        view_parts.append(f"topic: {topic}")
    if domain:
        view_parts.append(f"domain: {domain}")
    if summary:
        view_parts.append(f"summary: {summary}")
    if intent:
        view_parts.append(f"intent: {intent}")

    raw_keywords = semantic_focus_views.get("keywords", [])
    if isinstance(raw_keywords, list):
        cleaned_keywords = [
            _strip_nul(str(keyword)).strip()
            for keyword in raw_keywords
            if _strip_nul(str(keyword)).strip()
        ]
        if cleaned_keywords:
            view_parts.append(f"keywords: {', '.join(cleaned_keywords[:8])}")

    if expanded_context:
        view_parts.append(f"expanded: {expanded_context}")

    semantic_context_block = "\n".join(view_parts).strip()

    if semantic_context_block:
        return f"{base_text}\n\nSemantic focus: {focus_text}\n{semantic_context_block}".strip()

    if focus_text and focus_text != base_text:
        return f"{base_text}\n\nSemantic focus: {focus_text}".strip()

    return base_text


def _build_embedding_text_canonical(
    content_type: str,
    normalized_text: str,
    original_input: str,
    metadata: dict,
    semantic_focus_views: dict,
) -> str:
    base_content = _extract_canonical_content_for_embedding(
        content_type=content_type,
        normalized_text=normalized_text,
        original_input=original_input,
        metadata=metadata,
        semantic_focus_views=semantic_focus_views,
    )

    keywords = _extract_keywords_for_embedding(semantic_focus_views)
    if keywords:
        return f"{base_content}\n\nkeywords: {', '.join(keywords)}".strip()
    return base_content


def _extract_canonical_content_for_embedding(
    content_type: str,
    normalized_text: str,
    original_input: str,
    metadata: dict,
    semantic_focus_views: dict,
) -> str:
    def clean_text(value: object, max_chars: int = 1600) -> str:
        cleaned = _strip_nul(str(value or "")).strip()
        if not cleaned:
            return ""
        return cleaned[:max_chars]

    content_type_normalized = (content_type or "").strip().lower()

    if content_type_normalized in {"text", "audio"}:
        for candidate in [metadata.get("pre_focus_text"), original_input, normalized_text]:
            resolved = clean_text(candidate)
            if resolved:
                return resolved

    if content_type_normalized == "link":
        link_parts = [
            clean_text(metadata.get("summary"), max_chars=1200),
            clean_text(metadata.get("title"), max_chars=240),
        ]
        joined = "\n\n".join(part for part in link_parts if part)
        if joined:
            return joined
        for candidate in [metadata.get("pre_focus_text"), original_input, normalized_text]:
            resolved = clean_text(candidate)
            if resolved:
                return resolved

    if content_type_normalized == "youtube":
        youtube_parts = [
            clean_text(metadata.get("title"), max_chars=260),
            clean_text(metadata.get("description"), max_chars=900),
        ]
        tags_value = metadata.get("tags", [])
        if isinstance(tags_value, list):
            tags = ", ".join(str(tag).strip() for tag in tags_value if str(tag).strip())
            if tags:
                youtube_parts.append(clean_text(tags, max_chars=260))
        joined = "\n\n".join(part for part in youtube_parts if part)
        if joined:
            return joined
        for candidate in [metadata.get("pre_focus_text"), original_input, normalized_text]:
            resolved = clean_text(candidate)
            if resolved:
                return resolved

    if content_type_normalized == "file":
        file_parts = [
            clean_text(metadata.get("preview_text"), max_chars=1600),
            clean_text(metadata.get("file_title"), max_chars=240),
            clean_text(semantic_focus_views.get("summary", ""), max_chars=700) if isinstance(semantic_focus_views, dict) else "",
        ]
        joined = "\n\n".join(part for part in file_parts if part)
        if joined:
            return joined
        for candidate in [metadata.get("pre_focus_text"), original_input, normalized_text]:
            resolved = clean_text(candidate)
            if resolved:
                return resolved

    for candidate in [metadata.get("pre_focus_text"), original_input, normalized_text]:
        resolved = clean_text(candidate)
        if resolved:
            return resolved

    return clean_text(normalized_text)


def _extract_keywords_for_embedding(semantic_focus_views: dict) -> list[str]:
    raw_keywords = semantic_focus_views.get("keywords", []) if isinstance(semantic_focus_views, dict) else []
    if not isinstance(raw_keywords, list):
        return []

    clean_keywords: list[str] = []
    for keyword in raw_keywords:
        cleaned_keyword = _strip_nul(str(keyword)).strip().lower()
        if not cleaned_keyword:
            continue
        if cleaned_keyword in clean_keywords:
            continue
        clean_keywords.append(cleaned_keyword)
        if len(clean_keywords) >= 12:
            break
    return clean_keywords


def _strip_nul(value: str) -> str:
    return value.replace("\x00", " ")


def _sanitize_metadata(value):
    if isinstance(value, dict):
        return {k: _sanitize_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, str):
        return _strip_nul(value)
    return value
