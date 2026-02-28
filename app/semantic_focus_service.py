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
            "Extract a concise semantic focus from the text for clustering. "
            "Return topic/domain/summary entities using exact or near-exact phrases from input. "
            "Prefer one clear topic and one short domain context."
        )
        self.examples = [
            lx.data.ExampleData(
                text="Round-robin scheduling is a CPU process scheduling algorithm used in operating systems.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="topic",
                        extraction_text="round-robin scheduling",
                    ),
                    lx.data.Extraction(
                        extraction_class="domain",
                        extraction_text="operating systems",
                    ),
                    lx.data.Extraction(
                        extraction_class="summary",
                        extraction_text="cpu process scheduling algorithm",
                    ),
                ],
            )
        ]

    def build_focus(self, text: str, content_type: str) -> tuple[str, str]:
        cleaned_text = self._clean(text)
        if not cleaned_text:
            raise ValueError("Input vacío para semantic focus")
        
        if not self.enabled:
            raise ValueError("LANGEXTRACT_ENABLED está desactivado")
        langextract_focus = self._extract_with_langextract(cleaned_text, content_type)
        if langextract_focus:
            return langextract_focus, "langextract"

        raise ValueError("LangExtract no devolvió semantic focus")

    def _extract_with_langextract(self, text: str, content_type: str) -> str | None:
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
                try:
                    from langextract.providers import ollama

                    kwargs["resolver_params"] = {"format_handler": ollama.OLLAMA_FORMAT_HANDLER}
                except Exception:
                    pass
            result = lx.extract(**kwargs)
        except Exception as exc:
            print(f"Error during LangExtract semantic focus extraction: {exc}")
            return None

        parsed = self._parse_langextract_result(result)
        return self._shorten(parsed)

    def _parse_langextract_result(self, result: Any) -> str:
        extractions = getattr(result, "extractions", None)
        if not extractions:
            return ""

        topic = ""
        domain = ""
        summary = ""
        keywords: list[str] = []

        for extraction in extractions:
            extraction_class = str(getattr(extraction, "extraction_class", "")).strip().lower()
            extraction_text = self._clean(str(getattr(extraction, "extraction_text", "")).strip())
            if not extraction_text:
                continue

            if extraction_class in {"topic", "label", "category"} and not topic:
                topic = extraction_text
            elif extraction_class in {"domain", "context"} and not domain:
                domain = extraction_text
            elif extraction_class in {"summary", "description"} and not summary:
                summary = extraction_text
            elif extraction_class in {"keyword", "keywords", "term"}:
                keywords.append(extraction_text)

        if not topic and keywords:
            topic = keywords[0]

        parts = [part for part in [topic, domain, summary] if part]
        if not parts and keywords:
            parts = keywords[:3]

        return self._clean(". ".join(parts))

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
