import os
from pathlib import Path
from uuid import uuid4


FILE_STORAGE_DIR = os.getenv("FILE_STORAGE_DIR", "/app/uploads")


def ensure_storage_dir() -> Path:
    base_dir = Path(FILE_STORAGE_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def save_uploaded_file(file_bytes: bytes, original_filename: str) -> dict[str, str | int]:
    storage_dir = ensure_storage_dir()
    safe_name = Path(original_filename).name or "document.pdf"
    file_id = uuid4().hex
    stored_filename = f"{file_id}_{safe_name}"
    stored_path = storage_dir / stored_filename
    stored_path.write_bytes(file_bytes)

    return {
        "file_id": file_id,
        "original_filename": safe_name,
        "stored_filename": stored_filename,
        "stored_path": str(stored_path),
        "file_size_bytes": len(file_bytes),
    }
