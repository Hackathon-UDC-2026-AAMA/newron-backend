# API Docs

Este proyecto ya expone Swagger UI en tiempo de ejecución en:

- `/docs` (Swagger UI)
- `/redoc` (ReDoc)
- `/openapi.json` (esquema OpenAPI en vivo)

Para generar una copia versionable del esquema dentro de esta carpeta:

```bash
python -m app.export_openapi
```

En Windows PowerShell también puedes usar:

```powershell
python .\app\export_openapi.py
```

Se generará:

- `docs/openapi.json`
- `docs/swagger.html`
- `docs/redoc.html`
