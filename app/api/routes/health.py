from fastapi import APIRouter
from app.schemas import HealthCheck

router = APIRouter()


@router.get("/live", response_model=HealthCheck)
def is_alive():
    return {"status": "OK"}
