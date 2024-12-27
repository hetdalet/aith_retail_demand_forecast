from fastapi import APIRouter

from .routes import health
from .routes import ml_services
from .routes import tasks


api_router = APIRouter()
api_router.include_router(health.router, prefix="/health")
api_router.include_router(ml_services.router, prefix="/models")
api_router.include_router(tasks.router, prefix="/tasks")
