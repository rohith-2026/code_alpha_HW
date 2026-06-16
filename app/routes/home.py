from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.utils.constants import TEMPLATES_DIR


router = APIRouter()
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "home.html")
