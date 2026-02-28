import json
import re
from typing import Any
from urllib.parse import urlparse

import trafilatura
import yt_dlp

from app.classifier import extract_first_link_url, extract_first_youtube_url


def normalize_content(raw_input: str, content_type: str) -> tuple[str, dict]:
    clean_input = raw_input.strip()

    if content_type in {"text", "audio", "file"}:
        metadata = {"source": content_type, "length": len(clean_input)}
        return _prepare_for_embedding(clean_input), metadata

    if content_type == "youtube":
        youtube_url = extract_first_youtube_url(clean_input) or clean_input
        metadata = _extract_youtube_metadata(youtube_url)
        normalized_text = _prepare_for_embedding(metadata.get("title", ""))
        if not normalized_text:
            normalized_text = youtube_url
        return normalized_text, metadata

    link_url = extract_first_link_url(clean_input) or clean_input
    metadata = _extract_link_metadata(link_url)
    path_keywords = _extract_path_or_id(link_url).replace("-", " ")
    normalized_text = _prepare_for_embedding(" ".join(part for part in [metadata.get("title", ""), path_keywords] if part))
    if not normalized_text:
        normalized_text = _prepare_for_embedding(link_url)
    return normalized_text, metadata


def _extract_path_or_id(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path:
        return path.replace("/", "-")
    return "generic-resource"


def _extract_youtube_metadata(youtube_url: str) -> dict[str, Any]:
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(youtube_url, download=False) or {}
    except Exception:
        return {
            "source": "youtube",
            "url": youtube_url,
            "title": "",
            "description": "",
            "tags": [],
        }

    tags = info.get("tags") if isinstance(info.get("tags"), list) else []
    return {
        "source": "youtube",
        "url": info.get("webpage_url") or youtube_url,
        "title": (info.get("title") or "").strip(),
        "description": (info.get("description") or "").strip(),
        "tags": [str(tag) for tag in tags if tag][:20],
        "uploader": info.get("uploader") or info.get("channel") or "",
        "duration": info.get("duration"),
    }


def _extract_link_metadata(link_url: str) -> dict[str, Any]:
    parsed = urlparse(link_url)
    domain = parsed.netloc or "unknown-domain"
    fallback = {
        "source": "link",
        "url": link_url,
        "domain": domain,
        "title": "",
        "description": "",
        "summary": "",
    }

    try:
        downloaded = trafilatura.fetch_url(link_url)
        if not downloaded:
            return fallback

        extracted_json = trafilatura.extract(
            downloaded,
            output_format="json",
            include_comments=False,
            include_tables=False,
            deduplicate=True,
            favor_precision=True,
        )
        if not extracted_json:
            return fallback

        extracted = json.loads(extracted_json)
        title = _clean_text((extracted.get("title") or "").strip())
        full_text = _clean_text((extracted.get("text") or "").strip())
        summary = _build_compact_summary(full_text, max_sentences=6, max_chars=1400)
        description = summary[:320]

        return {
            "source": "link",
            "url": extracted.get("url") or link_url,
            "domain": domain,
            "title": title,
            "description": description,
            "summary": summary,
        }
    except Exception:
        return fallback


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _prepare_for_embedding(value: str) -> str:
    if not value:
        return ""

    with_spaces = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    with_spaces = re.sub(r"[-_/]", " ", with_spaces)
    normalized = _clean_text(with_spaces).lower()
    return normalized


def _build_compact_summary(text: str, max_sentences: int, max_chars: int) -> str:
    if not text:
        return ""

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    selected: list[str] = []
    current_length = 0

    for sentence in sentences:
        if len(sentence) < 30:
            continue
        proposed = current_length + len(sentence) + (1 if selected else 0)
        if proposed > max_chars:
            break
        selected.append(sentence)
        current_length = proposed
        if len(selected) >= max_sentences:
            break

    if not selected:
        return text[:max_chars].strip()
    return " ".join(selected).strip()


