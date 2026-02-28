import os
import re
from typing import Any

import langextract as lx


class SemanticFocusService:
    def __init__(self) -> None:
        self.enabled = os.getenv("LANGEXTRACT_ENABLED", "true").strip().lower() == "true"
        self.max_chars = int(os.getenv("SEMANTIC_FOCUS_MAX_CHARS", "240"))
        self.model_id = os.getenv("LANGEXTRACT_MODEL_ID", "llama3.2:3b")
        self.model_url = os.getenv("LANGEXTRACT_MODEL_URL", "http://ollama:11434")

        self.prompt_description = (
            "Reescribe el texto en una descripción corta y enriquecida semánticamente que preserve el tema principal "
            "y agregue contexto útil para agrupar documentos similares. "
            "Devuelve entidades para topic, domain, summary, keywords e intent. "
            "Responde siempre en español, de forma factual y concisa."
        )
        self.examples = [
            lx.data.ExampleData(
                text="Round-robin scheduling es un algoritmo de planificación de procesos de CPU usado en sistemas operativos.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="topic",
                        extraction_text="round-robin scheduling",
                    ),
                    lx.data.Extraction(
                        extraction_class="domain",
                        extraction_text="sistemas operativos",
                    ),
                    lx.data.Extraction(
                        extraction_class="summary",
                        extraction_text="algoritmo de planificación de procesos de CPU",
                    ),
                    lx.data.Extraction(
                        extraction_class="keywords",
                        extraction_text="CPU",
                    ),
                    lx.data.Extraction(
                        extraction_class="keywords",
                        extraction_text="planificación de procesos",
                    ),
                    lx.data.Extraction(
                        extraction_class="intent",
                        extraction_text="explicar su uso en sistemas operativos",
                    ),
                ],
            )
        ]

    def build_focus(self, text: str, content_type: str) -> tuple[str, str]:
        focus_views, source = self.build_focus_views(text, content_type)
        return self.compose_focus_text(focus_views), source

    def build_focus_views(self, text: str, content_type: str) -> tuple[dict[str, Any], str]:
        cleaned_text = self._clean(text)
        if not cleaned_text:
            raise ValueError("Input vacío para semantic focus")
        
        if not self.enabled:
            raise ValueError("LANGEXTRACT_ENABLED está desactivado")
        langextract_views = self._extract_with_langextract(cleaned_text, content_type)
        if langextract_views:
            return langextract_views, "langextract"

        raise ValueError("LangExtract no devolvió semantic focus")

    def _extract_with_langextract(self, text: str, content_type: str) -> dict[str, Any] | None:
        try:
            kwargs: dict[str, Any] = {
                "text_or_documents": text,
                "prompt_description": self.prompt_description,
                "examples": self.examples,
                "model_id": self.model_id,
                "show_progress": False,
                "max_char_buffer": 600,
                "extraction_passes": 1,
                "use_schema_constraints": False,
                "fence_output": False,
            }
            if self.model_url:
                kwargs["model_url"] = self.model_url
            result = lx.extract(**kwargs)
        except Exception as exc:
            print(f"Error during LangExtract semantic focus extraction: {exc}")
            return None

        parsed = self._parse_langextract_result(result)
        if not parsed:
            return None
        return parsed

    def _parse_langextract_result(self, result: Any) -> dict[str, Any]:
        extractions = getattr(result, "extractions", None)
        if not extractions:
            return {}

        topic = ""
        domain = ""
        summary = ""
        intent = ""
        keywords: list[str] = []

        for extraction in extractions:
            extraction_class = str(getattr(extraction, "extraction_class", "")).strip().lower()
            extraction_text_raw = self._normalize_extraction_text(getattr(extraction, "extraction_text", ""))
            if not extraction_text_raw:
                continue

            normalized_texts = [self._clean(fragment) for fragment in extraction_text_raw if self._clean(fragment)]
            if not normalized_texts:
                continue

            extraction_text = normalized_texts[0]

            if extraction_class in {"topic", "label", "category"} and not topic:
                topic = extraction_text
            elif extraction_class in {"domain", "context"} and not domain:
                domain = extraction_text
            elif extraction_class in {"summary", "description"} and not summary:
                summary = extraction_text
            elif extraction_class in {"intent", "goal", "purpose", "action"} and not intent:
                intent = extraction_text
            elif extraction_class in {"keyword", "keywords", "term"}:
                for text_item in normalized_texts:
                    for part in re.split(r"[,;]", text_item):
                        cleaned_part = self._clean(part)
                        if cleaned_part:
                            keywords.append(cleaned_part)

        if not topic and keywords:
            topic = keywords[0]

        keywords = list(dict.fromkeys(keywords))[:8]

        expanded_parts = [part for part in [topic, domain, summary, intent] if part]
        if not expanded_parts and keywords:
            expanded_parts = keywords[:3]

        expanded_context = self._shorten(self._clean(". ".join(expanded_parts)))

        return {
            "topic": self._shorten(topic),
            "domain": self._shorten(domain),
            "summary": self._shorten(summary),
            "keywords": keywords,
            "intent": self._shorten(intent),
            "expanded_context": expanded_context,
        }

    def compose_focus_text(self, focus_views: dict[str, Any]) -> str:
        topic = self._clean(str(focus_views.get("topic", "")))
        domain = self._clean(str(focus_views.get("domain", "")))
        summary = self._clean(str(focus_views.get("summary", "")))
        intent = self._clean(str(focus_views.get("intent", "")))
        expanded_context = self._clean(str(focus_views.get("expanded_context", "")))

        raw_keywords = focus_views.get("keywords", [])
        if not isinstance(raw_keywords, list):
            raw_keywords = []
        keywords = [self._clean(str(keyword)) for keyword in raw_keywords if self._clean(str(keyword))]

        parts = [
            f"topic: {topic}" if topic else "",
            f"domain: {domain}" if domain else "",
            f"summary: {summary}" if summary else "",
            f"intent: {intent}" if intent else "",
            f"keywords: {', '.join(keywords[:8])}" if keywords else "",
            f"expanded: {expanded_context}" if expanded_context else "",
        ]

        return self._shorten(self._clean(" | ".join(part for part in parts if part)))

    def _shorten(self, text: str) -> str:
        cleaned_text = self._clean(text)
        if len(cleaned_text) <= self.max_chars:
            return cleaned_text

        truncated = cleaned_text[: self.max_chars]
        split_index = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if split_index >= 80:
            return truncated[: split_index + 1].strip()
        return truncated.strip()

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_extraction_text(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, (str, int, float)):
            return [str(value)]

        if isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                if isinstance(item, (str, int, float)):
                    normalized.append(str(item))
            return normalized

        return []
