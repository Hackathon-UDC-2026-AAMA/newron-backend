import json
import os
import re
from collections import Counter
from urllib import request


class LlmCategoryService:
    def __init__(self) -> None:
        self.enabled = os.getenv("CATEGORY_NAME_LLM_ENABLED", "true").strip().lower() == "true"
        self.model_id = os.getenv("CATEGORY_NAME_LLM_MODEL_ID", os.getenv("LANGEXTRACT_MODEL_ID", "llama3.2:3b"))
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", os.getenv("LANGEXTRACT_MODEL_URL", "http://ollama:11434")).rstrip("/")
        self.timeout_seconds = int(os.getenv("CATEGORY_NAME_LLM_TIMEOUT", "25"))

    def generate_category(self, keywords: list[str], sample_texts: list[str], fallback_label: str) -> tuple[str, str | None]:
        fallback_description = self._build_fallback_description(keywords, sample_texts, fallback_label)

        if not self.enabled:
            return fallback_label, fallback_description

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
                return category_name, category_description or fallback_description
            return fallback_label, fallback_description
        except Exception:
            return fallback_label, fallback_description

    def _build_prompt(self, keywords: list[str], sample_texts: list[str]) -> str:
        sample_snippets = [text.strip() for text in sample_texts if text and text.strip()][:8]
        snippets_block = "\n".join(f"- {snippet[:260]}" for snippet in sample_snippets)
        keyword_block = ", ".join(keywords) if keywords else "(sin keywords explícitas)"

        return (
            "Eres un clasificador semántico para nombrar categorías de contenido.\n"
            "Devuelve SOLO JSON válido con esta forma exacta: {\"category_name\":\"...\",\"category_description\":\"...\"}.\n"
            "Reglas: category_name en español, 2-5 palabras, sin barras '/', sin dos puntos, sin comillas extra.\n"
            "Reglas: category_description en español, 1 frase breve (max 140 caracteres), explicando por qué cae en esa categoría.\n"
            "Objetivo: generar un título que represente el tema común de TODO el cluster, no de un solo elemento.\n"
            "Si hay variaciones entre entradas, prioriza el denominador común.\n"
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

    def _build_fallback_description(self, keywords: list[str], sample_texts: list[str], fallback_label: str) -> str:
        clean_keywords = [self._normalize_keyword(keyword) for keyword in keywords if self._normalize_keyword(keyword)]
        clean_keywords = list(dict.fromkeys(clean_keywords))[:4]

        if not clean_keywords:
            clean_keywords = self._extract_terms_from_samples(sample_texts)[:4]

        if clean_keywords:
            description = f"Contenido relacionado con {', '.join(clean_keywords)}."
        else:
            clean_label = " ".join(str(fallback_label or "contenido relacionado").split()).strip().lower()
            if not clean_label:
                clean_label = "contenido relacionado"
            description = f"Contenido relacionado con {clean_label}."

        if len(description) > 140:
            description = description[:140].rstrip()
        return description

    def _extract_terms_from_samples(self, sample_texts: list[str]) -> list[str]:
        stop_words = {
            "de",
            "la",
            "el",
            "en",
            "y",
            "a",
            "que",
            "los",
            "las",
            "un",
            "una",
            "por",
            "para",
            "con",
            "del",
            "sobre",
            "como",
            "desde",
            "hasta",
            "this",
            "that",
            "with",
            "from",
            "and",
            "the",
        }

        frequencies = Counter()
        for sample in sample_texts[:10]:
            for token in re.findall(r"[a-záéíóúñü0-9]{4,}", str(sample).lower()):
                if token in stop_words:
                    continue
                if token.isdigit():
                    continue
                frequencies[token] += 1

        return [token for token, _ in frequencies.most_common(8)]

    def _normalize_keyword(self, value: object) -> str:
        return " ".join(str(value or "").strip().lower().split())
