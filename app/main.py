from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import home, pages, predict
from app.utils.constants import APP_NAME, STATIC_DIR


app = FastAPI(title=APP_NAME)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(home.router)
app.include_router(pages.router)
app.include_router(predict.router, prefix="/api")

