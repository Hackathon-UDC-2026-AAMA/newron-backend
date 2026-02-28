import os
from typing import Any

import langextract as lx


class LlmClassificationService:
    def __init__(self) -> None:
        self.enabled = os.getenv("LLM_CLASSIFICATION_ENABLED", "true").strip().lower() == "true"
        self.model_id = os.getenv("LANGEXTRACT_MODEL_ID", "llama3.2:3b")
        self.model_url = os.getenv("LANGEXTRACT_MODEL_URL", "http://ollama:11434")

        self.prompt_description = (
            "Classify the input into exactly one content type: text, link, or youtube. "
            "Return one extraction with class 'classification', extraction_text copied from input, "
            "and attribute 'content_type' set to one of [text, link, youtube]."
        )
        self.examples = [
            lx.data.ExampleData(
                text="https://en.wikipedia.org/wiki/Round-robin_scheduling",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="classification",
                        extraction_text="https://en.wikipedia.org/wiki/Round-robin_scheduling",
                        attributes={"content_type": "link"},
                    )
                ],
            ),
            lx.data.ExampleData(
                text="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="classification",
                        extraction_text="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        attributes={"content_type": "youtube"},
                    )
                ],
            ),
            lx.data.ExampleData(
                text="Tengo que estudiar round robin para el examen",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="classification",
                        extraction_text="Tengo que estudiar round robin para el examen",
                        attributes={"content_type": "text"},
                    )
                ],
            ),
        ]

    def classify(self, raw_input: str) -> tuple[str, str]:
        candidate = self._classify_with_langextract(raw_input)
        if candidate in {"text", "link", "youtube"}:
            return candidate, "langextract"

        raise ValueError("LangExtract no devolvió content_type válido")

    def _classify_with_langextract(self, raw_input: str) -> str | None:
        try:
            kwargs: dict[str, Any] = {
                "text_or_documents": raw_input,
                "prompt_description": self.prompt_description,
                "examples": self.examples,
                "model_id": self.model_id,
                "show_progress": False,
                "max_char_buffer": 500,
                "extraction_passes": 1,
                "use_schema_constraints": False,
                "fence_output": False,
            }
            print(f"LLM Classification model_url: {self.model_url}")
            if self.model_url:
                kwargs["model_url"] = self.model_url
                try:
                    from langextract.providers import ollama

                    kwargs["resolver_params"] = {"format_handler": ollama.OLLAMA_FORMAT_HANDLER}
                except Exception:
                    pass

            print(f"LLM Classification kwargs: {kwargs}")
            result = lx.extract(**kwargs)
            print(f"LLM Classification result: {result}")
        except Exception:
            return None

        extractions = getattr(result, "extractions", None)
        if not extractions:
            return None

        for extraction in extractions:
            attributes = getattr(extraction, "attributes", None)
            if isinstance(attributes, dict):
                content_type = str(attributes.get("content_type", "")).strip().lower()
                if content_type in {"text", "link", "youtube"}:
                    return content_type

            extraction_class = str(getattr(extraction, "extraction_class", "")).strip().lower()
            extraction_text = str(getattr(extraction, "extraction_text", "")).strip().lower()
            if extraction_class in {"content_type", "type", "classification"} and extraction_text in {"text", "link", "youtube"}:
                return extraction_text

        return None
