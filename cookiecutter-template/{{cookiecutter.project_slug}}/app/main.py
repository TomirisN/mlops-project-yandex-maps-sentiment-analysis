from fastapi import FastAPI
from prometheus_client import make_asgi_app

app = FastAPI(
    title="{{ cookiecutter.project_name }}",
    description="{{ cookiecutter.description }}",
    version="0.1.0",
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
def root():
    return {
        "project": "{{ cookiecutter.project_slug }}",
        "status": "ok",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": False}


@app.get("/api/v1/health")
def api_health():
    return {"status": "healthy", "model_loaded": False}
