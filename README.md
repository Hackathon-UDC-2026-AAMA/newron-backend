# Newron Backend

Backend de ingesta de archivos y clustering semántico (FastAPI + PostgreSQL + Ollama) para indexar contenido y agruparlo en clusters evolutivos.

## Requisitos

- Docker + Docker Compose
- Recursos de CPU/RAM suficientes para embeddings y STT
- Modelo LLM disponible en Ollama cuando se use nombrado semántico de categorías

## Despliegue

Desde la raíz del proyecto:

```bash
docker compose up --build
```

Notas operativas:

- El backend arranca en `http://localhost:8000`.
- Swagger runtime: `http://localhost:8000/docs`.
- Si el modelo LLM configurado no está instalado en el contenedor de Ollama, pueden fallar warmup/clasificación por LLM (el backend intenta continuar con fallback cuando aplica).

## Flujo general de indexación

1. **Entrada**: texto, link, YouTube, audio o archivo.
2. **Normalización por tipo**:
   - `text/audio`: limpieza textual.
   - `link`: extracción de metadata/resumen.
   - `youtube`: título/descripcion/tags.
   - `file`: extracción de texto por extensión (incluye markdown y archivos de código).
3. **Representación para embedding**:
   - Con LangExtract: enfoque semántico estructurado.
   - Sin LangExtract: contenido canónico limpio + keywords fallback.
4. **Embedding** con `sentence-transformers`.
5. **Asignación de cluster** por similitud coseno:
   - umbral por tipo de contenido,
   - guard temático para texto/audio,
   - umbral adaptativo por cluster (cohesión interna).
6. **Actualización de estado del cluster**:
   - Recuento real de items del cluster (`size`).
   - Recomputo del **centroide** como promedio de embeddings de todos los items actuales.
   - Recomputo de **keywords del cluster** combinando términos comunes + TF-IDF.
   - Recomputo de **título y descripción** de categoría (LLM con fallback determinista).
   - Si el cluster queda vacío (por movimientos/fusiones), se elimina.
7. **Unificación de clusters** por overlap de keywords comunes.

### Detalle: actualización de estado del cluster

Cada vez que se inserta un item (o se mueve manualmente entre clusters), el backend ejecuta un recálculo para mantener coherencia del cluster:

1. **Carga todos los items del cluster** desde base de datos.
2. **Si no quedan items**, elimina el cluster para evitar clusters huérfanos.
3. **Recalcula `size`** con el número real de items.
4. **Recalcula `centroid`** promediando los embeddings actuales (no usa un valor histórico stale).
5. **Recalcula metadata semántica**:
   - `cluster_keywords`: términos representativos del conjunto,
   - `cluster_label`: título de categoría,
   - `cluster_description`: descripción breve de categoría.
6. **Intenta fusión entre clusters** por overlap de keywords; si fusiona, vuelve a recalcular para dejar el estado final consistente.

Esto garantiza que, aunque entren datos nuevos o se haga re-clustering manual, el cluster refleje siempre su contenido real en ese momento.

## Modos de funcionamiento

### 1) Modo semántico (con LangExtract)

- Más contexto semántico por item (`topic/domain/summary/intent/keywords`).
- Mejor comportamiento en entradas ambiguas y similares.
- Mayor latencia y dependencia del modelo.

### 2) Modo determinista (sin LangExtract)

- Más rápido y estable.
- Keywords por frecuencia + contenido canónico por tipo.
- Menor coste y menos dependencias externas.
- Calidad generalmente buena con umbrales bien ajustados (complicado de ajustar umbrales).

## Variables de entorno principales

- Conectividad DB: `DB_*`, `POSTGRES_*`
- Umbrales clustering: `SIMILARITY_THRESHOLD`, `TEXT_*`, `LINK_*`, `YOUTUBE_*`, `FILE_*`
- Guard temático texto/audio: `TEXT_THEME_*`
- Embeddings/STT: `EMBEDDING_MODEL`, `STT_*`
- LangExtract: `LANGEXTRACT_ENABLED`, `LANGEXTRACT_MODEL_ID`, `LANGEXTRACT_MODEL_URL`
- LLM de categoría: `CATEGORY_NAME_LLM_*`, `OLLAMA_BASE_URL`

## Documentación OpenAPI exportada

En arranque de contenedor se exporta automáticamente a `docs/`:

- `docs/openapi.json`
- `docs/swagger.html`
- `docs/redoc.html`

También puede ejecutarse manualmente:

```bash
python -m app.export_openapi
```

## Cola simple de ingesta (BackgroundTasks)

Para evitar que peticiones de ingesta largas bloqueen el proceso principal, se implementó una cola simple en segundo plano usando `BackgroundTasks` de FastAPI.

- Los endpoints de ingesta aceptan la petición y responden rápido con estado `processing`.
- El trabajo pesado (embedding, clustering, recalculo de cluster, etc.) se ejecuta después en background.
- Esto permite que los endpoints `GET` sigan respondiendo mientras se procesan ingestas.

Objetivo de esta decisión:

- no inhabilitar el backend por operaciones de ingestión costosas,
- mantener buena experiencia de usuario en consultas,
- evitar introducir infraestructura extra (broker/worker) en esta fase.

Limitación conocida:

- si el proceso del backend cae, las tareas en memoria en curso se pierden.
