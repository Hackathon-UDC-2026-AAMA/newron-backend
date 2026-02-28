import os
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


MAX_EXTRACTED_CHARS = 50000
DEFAULT_FILE_TITLE_WEIGHT = int(os.getenv("FILE_TITLE_WEIGHT", "4"))


def extract_text_from_file(file_bytes: bytes, extension: str) -> str:
    normalized_extension = extension.lower()

    if normalized_extension == ".pdf":
        return _extract_pdf_text(file_bytes)

    if normalized_extension in {".txt", ".md", ".markdown"}:
        text = file_bytes.decode("utf-8", errors="ignore")
        return _clean_limit(text)

    raise ValueError("Formato de archivo no soportado")


def extract_title_from_file(file_bytes: bytes, extension: str, filename: str) -> str:
    normalized_extension = extension.lower()
    filename_title = _title_from_filename(filename)

    if normalized_extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(file_bytes))
            metadata_title = _clean_title(str(getattr(reader.metadata, "title", "") or ""))
            if metadata_title:
                return metadata_title

            for page in reader.pages[:2]:
                page_text = page.extract_text() or ""
                for line in page_text.splitlines():
                    maybe_title = _clean_title(line)
                    if len(maybe_title.split()) >= 4:
                        return maybe_title
        except Exception:
            return filename_title

    if normalized_extension in {".txt", ".md", ".markdown"}:
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            for line in text.splitlines()[:20]:
                maybe_title = _clean_title(line)
                if len(maybe_title.split()) >= 3:
                    return maybe_title
        except Exception:
            return filename_title

    return filename_title


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_text.append(page_text)

    return _clean_limit("\n".join(pages_text))


def _clean_limit(text: str) -> str:
    without_nul = text.replace("\x00", " ")
    normalized = " ".join(without_nul.split())
    return normalized[:MAX_EXTRACTED_CHARS].strip()


def build_index_text_for_clustering(
    text: str,
    title: str | None = None,
    title_weight: int = DEFAULT_FILE_TITLE_WEIGHT,
    chunk_words: int = 280,
    max_chunks: int = 8,
) -> str:
    words = text.split()
    if not words:
        return ""

    chunks: list[str] = []
    start = 0
    while start < len(words) and len(chunks) < max_chunks:
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end

    sampled_chunks: list[str] = []
    for chunk in chunks:
        sampled_chunks.append(" ".join(chunk.split()[:90]))

    body = "\n\n".join(sampled_chunks).strip()
    clean_title = _clean_title(title or "")

    if not clean_title:
        return body

    safe_weight = max(1, min(int(title_weight), 12))
    weighted_title = "\n".join([clean_title] * safe_weight)
    return f"{weighted_title}\n\n{body}".strip()


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    normalized = stem.replace("_", " ").replace("-", " ")
    return _clean_title(normalized) or "Untitled Document"


def _clean_title(value: str) -> str:
    return _clean_limit(value)[:220]
