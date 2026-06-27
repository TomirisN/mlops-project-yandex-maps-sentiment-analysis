from fastapi import FastAPI

app = FastAPI(title="{{ cookiecutter.project_name }}")

@app.get("/")
def root():
    return {"project": "{{ cookiecutter.project_slug }}", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}
