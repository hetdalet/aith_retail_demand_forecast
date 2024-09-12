import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# from app.log import get_logger
from app.api import api_router


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
current_dir = os.path.dirname(__file__)
static_dir = os.path.join(current_dir, "view", "templates", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# logger = get_logger(name=__name__, level=logging.INFO)
app.include_router(api_router)
