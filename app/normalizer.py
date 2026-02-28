from urllib.parse import urlparse

from app.classifier import extract_first_link_url, extract_first_youtube_url


def normalize_content(raw_input: str, content_type: str) -> tuple[str, dict]:
    clean_input = raw_input.strip()

    if content_type == "text":
        metadata = {"source": "text", "length": len(clean_input)}
        return clean_input, metadata

    if content_type == "youtube":
        youtube_url = extract_first_youtube_url(clean_input) or clean_input
        video_id = _extract_path_or_id(youtube_url)
        metadata = {
            "source": "youtube",
            "url": youtube_url,
            "title": f"Demo title for video {video_id}",
            "description": f"Simulated description for YouTube resource {video_id}.",
            "tags": ["demo", "youtube", "simulated"],
        }
        normalized_text = " ".join([metadata["title"], metadata["description"], " ".join(metadata["tags"])])
        return normalized_text, metadata

    link_url = extract_first_link_url(clean_input) or clean_input
    parsed = urlparse(link_url)
    domain = parsed.netloc or "unknown-domain"
    page_slug = _extract_path_or_id(link_url)
    metadata = {
        "source": "link",
        "url": link_url,
        "title": f"Simulated article from {domain}",
        "description": f"Auto-generated summary placeholder for path {page_slug}.",
    }
    normalized_text = f"{metadata['title']} {metadata['description']}"
    return normalized_text, metadata


def _extract_path_or_id(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path:
        return path.replace("/", "-")
    return "generic-resource"
