import re
from urllib.parse import parse_qs, urlparse

YOUTUBE_REGEX = re.compile(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/\S+", re.IGNORECASE)
LINK_REGEX = re.compile(r"https?://\S+", re.IGNORECASE)


def classify_input(raw_input: str) -> str:
    value = raw_input.strip()
    if YOUTUBE_REGEX.search(value):
        return "youtube"
    if LINK_REGEX.search(value):
        return "link"
    return "text"


def extract_first_youtube_url(raw_input: str) -> str | None:
    match = YOUTUBE_REGEX.search(raw_input)
    if not match:
        return None
    return normalize_url(_clean_url_token(match.group(0)))


def extract_first_link_url(raw_input: str) -> str | None:
    match = LINK_REGEX.search(raw_input)
    if not match:
        return None
    return normalize_url(_clean_url_token(match.group(0)))


def _clean_url_token(url: str) -> str:
    return url.rstrip(".,;:!?)\"]}'")


def normalize_url(url: str) -> str:
    parsed = urlparse(url)

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    if netloc in {"youtu.be", "youtube.com", "m.youtube.com"}:
        video_id = _extract_youtube_video_id(netloc, parsed)
        if video_id:
            return f"https://youtube.com/watch?v={video_id}"

    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if not path:
        path = "/"

    query = parsed.query
    if parsed.fragment:
        query = f"{query}&fragment={parsed.fragment}" if query else f"fragment={parsed.fragment}"

    normalized = f"{scheme}://{netloc}{path}"
    if query:
        normalized = f"{normalized}?{query}"
    return normalized


def _extract_youtube_video_id(netloc: str, parsed_url) -> str | None:
    if netloc == "youtu.be":
        return parsed_url.path.strip("/") or None

    query_params = parse_qs(parsed_url.query)
    if "v" in query_params and query_params["v"]:
        return query_params["v"][0]

    path_parts = [part for part in parsed_url.path.strip("/").split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "v"}:
        return path_parts[1]
    return None
