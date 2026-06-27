from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import config

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/ui", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "mlflow_url": config.MLFLOW_UI_URL,
            "page": "dashboard",
        },
    )


@router.get("/ui/inference", response_class=HTMLResponse)
async def ui_inference(request: Request):
    return templates.TemplateResponse(
        request,
        "inference.html",
        {
            "mlflow_url": config.MLFLOW_UI_URL,
            "page": "inference",
        },
    )


@router.get("/ui/predictions", response_class=HTMLResponse)
async def ui_predictions(request: Request):
    return templates.TemplateResponse(
        request,
        "predictions.html",
        {
            "mlflow_url": config.MLFLOW_UI_URL,
            "page": "predictions",
        },
    )


@router.get("/ui/experiments", response_class=HTMLResponse)
async def ui_experiments(request: Request):
    return templates.TemplateResponse(
        request,
        "experiments.html",
        {
            "mlflow_url": config.MLFLOW_UI_URL,
            "page": "experiments",
        },
    )
