from decimal import Decimal

from sqlalchemy.orm import Session

from app.schemas import MLService
from app.models import MLService as MLServiceModelDB
from app import repository


def get_service(name: str, db: Session) -> MLService:
    srv = repository.read_by_id(
        MLServiceModelDB,
        name,
        MLService,
        db,
        id_field="name"
    )
    return srv
