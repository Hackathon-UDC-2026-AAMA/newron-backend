import json
import os
from urllib import request

from sentence_transformers import SentenceTransformer


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def warmup_embedding_model() -> None:
    model_id = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
    print(f"[warmup] Loading embedding model: {model_id}")
    SentenceTransformer(model_id)
    print("[warmup] Embedding model ready")


def pull_ollama_model(base_url: str, model_id: str, timeout_seconds: int) -> None:
    if not model_id:
        return

    payload = {"name": model_id, "stream": False}
    req = request.Request(
        url=f"{base_url.rstrip('/')}/api/pull",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        response.read()


def ollama_model_exists(base_url: str, model_id: str, timeout_seconds: int) -> bool:
    req = request.Request(
        url=f"{base_url.rstrip('/')}/api/tags",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")

    payload = json.loads(body)
    models = payload.get("models", []) if isinstance(payload, dict) else []
    normalized_target = model_id.strip().lower()

    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name", "")).strip().lower()
        if name == normalized_target:
            return True
        if name.startswith(f"{normalized_target}:"):
            return True
    return False


def warmup_ollama_models() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", os.getenv("LANGEXTRACT_MODEL_URL", "http://ollama:11434"))
    timeout_seconds = int(os.getenv("OLLAMA_PULL_TIMEOUT_SECONDS", "240"))

    models: list[str] = []
    if _env_flag("LANGEXTRACT_ENABLED", True):
        models.append(os.getenv("LANGEXTRACT_MODEL_ID", "").strip())
    if _env_flag("CATEGORY_NAME_LLM_ENABLED", True):
        models.append(os.getenv("CATEGORY_NAME_LLM_MODEL_ID", "").strip())

    unique_models = [model for model in dict.fromkeys(models) if model]
    if not unique_models:
        print("[warmup] No Ollama models configured for warmup")
        return

    for model_id in unique_models:
        try:
            if ollama_model_exists(base_url, model_id, timeout_seconds):
                print(f"[warmup] Ollama model already present: {model_id}")
                continue
            print(f"[warmup] Pulling Ollama model: {model_id}")
            pull_ollama_model(base_url, model_id, timeout_seconds)
            print(f"[warmup] Ollama model ready: {model_id}")
        except Exception as exc:
            print(f"[warmup] Warning: could not pull {model_id}: {exc}")


def main() -> None:
    warmup_embedding_model()
    warmup_ollama_models()


if __name__ == "__main__":
    main()
