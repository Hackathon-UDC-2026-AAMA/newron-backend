import json
import os
from urllib import request


class LlmCategoryService:
    def __init__(self) -> None:
        self.enabled = os.getenv("CATEGORY_NAME_LLM_ENABLED", "true").strip().lower() == "true"
        self.model_id = os.getenv("CATEGORY_NAME_LLM_MODEL_ID", os.getenv("LANGEXTRACT_MODEL_ID", "llama3.2:3b"))
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", os.getenv("LANGEXTRACT_MODEL_URL", "http://ollama:11434")).rstrip("/")
        self.timeout_seconds = int(os.getenv("CATEGORY_NAME_LLM_TIMEOUT", "25"))

    def generate_name(self, keywords: list[str], sample_texts: list[str], fallback_label: str) -> str:
        if not self.enabled:
            return fallback_label

        if not keywords:
            return fallback_label

        prompt = self._build_prompt(keywords, sample_texts)
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        try:
            req = request.Request(
                url=f"{self.ollama_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            model_output = str(parsed.get("response", "")).strip()
            category_name = self._parse_category_name(model_output)
            if category_name:
                return category_name
            return fallback_label
        except Exception:
            return fallback_label

    def _build_prompt(self, keywords: list[str], sample_texts: list[str]) -> str:
        sample_snippets = [text.strip() for text in sample_texts if text and text.strip()][:3]
        snippets_block = "\n".join(f"- {snippet[:180]}" for snippet in sample_snippets)
        keyword_block = ", ".join(keywords)

        return (
            "Eres un clasificador semántico para nombrar categorías de contenido.\n"
            "Devuelve SOLO JSON válido con esta forma exacta: {\"category_name\":\"...\"}.\n"
            "Reglas: category_name en español, 2-5 palabras, sin barras '/', sin dos puntos, sin comillas extra.\n"
            f"Keywords TF-IDF: {keyword_block}\n"
            f"Ejemplos del cluster:\n{snippets_block}\n"
        )

    def _parse_category_name(self, model_output: str) -> str:
        if not model_output:
            return ""

        try:
            parsed = json.loads(model_output)
            value = parsed.get("category_name")
            if isinstance(value, str):
                return self._sanitize(value)
        except Exception:
            pass

        return self._sanitize(model_output)

    def _sanitize(self, value: str) -> str:
        cleaned = value.strip().strip('"').replace("/", " ")
        cleaned = " ".join(cleaned.split())
        words = cleaned.split()
        if len(words) > 5:
            cleaned = " ".join(words[:5])
        return cleaned
