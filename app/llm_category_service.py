import json
import os
import re
from urllib import request


class LlmCategoryService:
    def __init__(self) -> None:
        self.enabled = os.getenv("CATEGORY_NAME_LLM_ENABLED", "true").strip().lower() == "true"
        self.model_id = os.getenv("CATEGORY_NAME_LLM_MODEL_ID", os.getenv("LANGEXTRACT_MODEL_ID", "llama3.2:3b"))
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", os.getenv("LANGEXTRACT_MODEL_URL", "http://ollama:11434")).rstrip("/")
        self.timeout_seconds = int(os.getenv("CATEGORY_NAME_LLM_TIMEOUT", "25"))

    def generate_category(self, keywords: list[str], sample_texts: list[str], fallback_label: str) -> tuple[str, str | None]:
        if not self.enabled:
            return fallback_label, None

        if not keywords:
            return fallback_label, None

        prompt = self._build_prompt(keywords, sample_texts)
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        try:
            req = request.Request(
                url=f"{self.ollama_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            model_output = str(parsed.get("response", "")).strip()

            if not model_output:
                message = parsed.get("message")
                if isinstance(message, dict):
                    model_output = str(message.get("content", "")).strip()

            category_name, category_description = self._parse_category_payload(model_output)
            if category_name:
                return category_name, category_description
            return fallback_label, None
        except Exception:
            return fallback_label, None

    def _build_prompt(self, keywords: list[str], sample_texts: list[str]) -> str:
        sample_snippets = [text.strip() for text in sample_texts if text and text.strip()][:3]
        snippets_block = "\n".join(f"- {snippet[:180]}" for snippet in sample_snippets)
        keyword_block = ", ".join(keywords)

        return (
            "Eres un clasificador semántico para nombrar categorías de contenido.\n"
            "Devuelve SOLO JSON válido con esta forma exacta: {\"category_name\":\"...\",\"category_description\":\"...\"}.\n"
            "Reglas: category_name en español, 2-5 palabras, sin barras '/', sin dos puntos, sin comillas extra.\n"
            "Reglas: category_description en español, 1 frase breve (max 140 caracteres), explicando por qué cae en esa categoría, intenta tener en cuenta todas las entradas a la hora de generar la categoría.\n"
            f"Keywords TF-IDF: {keyword_block}\n"
            f"Ejemplos del cluster:\n{snippets_block}\n"
        )

    def _parse_category_payload(self, model_output: str) -> tuple[str, str | None]:
        if not model_output:
            return "", None

        parsed = self._parse_json_like(model_output)
        if isinstance(parsed, dict):
            raw_name = parsed.get("category_name") or parsed.get("name") or parsed.get("title")
            raw_description = parsed.get("category_description") or parsed.get("description")
            if isinstance(raw_name, str):
                return self._sanitize_name(raw_name), self._sanitize_description(raw_description)

            if len(parsed) == 1:
                key, value = next(iter(parsed.items()))
                if isinstance(key, str):
                    return self._sanitize_name(key), self._sanitize_description(value)

        if isinstance(parsed, str):
            reparsed = self._parse_json_like(parsed)
            if isinstance(reparsed, dict) and len(reparsed) == 1:
                key, value = next(iter(reparsed.items()))
                if isinstance(key, str):
                    return self._sanitize_name(key), self._sanitize_description(value)

        return self._sanitize_name(model_output), None

    def _parse_json_like(self, value: str) -> object | None:
        text = value.strip()
        if not text:
            return None

        candidates = [text]

        unfenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        if unfenced != text:
            candidates.append(unfenced)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                continue

        return None

    def _sanitize_name(self, value: str) -> str:
        cleaned = value.strip().strip('"').replace("/", " ")

        dict_like_key = re.search(r"[\{,]\s*[\"']?([^\"':\{\}]{2,80})[\"']?\s*:", cleaned)
        if dict_like_key:
            cleaned = dict_like_key.group(1)

        cleaned = " ".join(cleaned.split())
        words = cleaned.split()
        if len(words) > 5:
            cleaned = " ".join(words[:5])
        return cleaned

    def _sanitize_description(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None

        cleaned = " ".join(value.strip().strip('"').split())
        if not cleaned:
            return None
        if len(cleaned) > 140:
            cleaned = cleaned[:140].rstrip()
        return cleaned
