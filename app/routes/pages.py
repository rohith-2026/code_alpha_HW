from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.utils.constants import TEMPLATES_DIR


router = APIRouter()
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@router.get("/draw")
async def draw_page(request: Request):
    return templates.TemplateResponse(request, "draw.html")


@router.get("/process")
async def process_page(request: Request):
    return templates.TemplateResponse(request, "process.html")


@router.get("/result")
async def result_page(
    request: Request,
    prediction: str = "",
    confidence: float = 0.0,
    alternatives: str = "",
):
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "prediction": prediction,
            "confidence": confidence,
            "alternatives": alternatives,
        },
    )
