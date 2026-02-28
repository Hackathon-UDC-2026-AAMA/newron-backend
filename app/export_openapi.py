import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

from app.main import app


def export_openapi() -> tuple[Path, Path, Path]:
        docs_dir = Path(__file__).resolve().parents[1] / "docs"
        output_path = docs_dir / "openapi.json"
        swagger_html_path = docs_dir / "swagger.html"
        redoc_html_path = docs_dir / "redoc.html"
        docs_dir.mkdir(parents=True, exist_ok=True)

        schema = app.openapi()
        schema_json_pretty = json.dumps(schema, ensure_ascii=False, indent=2)
        schema_json_inline = json.dumps(schema, ensure_ascii=False).replace("</", "<\\/")

        output_path.write_text(
                schema_json_pretty,
                encoding="utf-8",
        )

        swagger_html_path.write_text(
                _build_swagger_html(schema_json_inline),
                encoding="utf-8",
        )
        redoc_html_path.write_text(
                _build_redoc_html(schema_json_inline),
                encoding="utf-8",
        )

        return output_path, swagger_html_path, redoc_html_path


def _build_swagger_html(schema_json_inline: str) -> str:
        return f"""<!doctype html>
<html lang=\"es\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Swagger UI - Semantic Ingestion Backend</title>
        <link rel=\"stylesheet\" href=\"https://unpkg.com/swagger-ui-dist@5/swagger-ui.css\" />
        <style>body{{margin:0;background:#fafafa}}</style>
    </head>
    <body>
        <div id=\"swagger-ui\"></div>
        <script src=\"https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js\"></script>
        <script>
            const spec = {schema_json_inline};
            window.ui = SwaggerUIBundle({{
                spec,
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [SwaggerUIBundle.presets.apis],
            }});
        </script>
    </body>
</html>
"""


def _build_redoc_html(schema_json_inline: str) -> str:
        return f"""<!doctype html>
<html lang=\"es\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>ReDoc - Semantic Ingestion Backend</title>
        <style>body{{margin:0;padding:0}}</style>
    </head>
    <body>
        <div id=\"redoc-container\"></div>
        <script src=\"https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js\"></script>
        <script>
            const spec = {schema_json_inline};
            Redoc.init(spec, {{}}, document.getElementById('redoc-container'));
        </script>
    </body>
</html>
"""


if __name__ == "__main__":
        openapi_path, swagger_path, redoc_path = export_openapi()
        print(f"OpenAPI exportado en: {openapi_path}")
        print(f"Swagger HTML generado en: {swagger_path}")
        print(f"ReDoc HTML generado en: {redoc_path}")
