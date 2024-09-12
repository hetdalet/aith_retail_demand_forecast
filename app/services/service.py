from decimal import Decimal

from sqlalchemy.orm import Session

from app.domain_types import ServicePricingType
from app.schemas import Service
from app.models import Service as ServiceDB
from app import repository


def get_service(name: str, db: Session) -> Service:
    srv = repository.read_by_id(
        ServiceDB,
        name,
        Service,
        db,
        id_field="name"
    )
    return srv


def calc_service_price(service: Service, input_data) -> Decimal:
    if service.pricing_type == ServicePricingType.fixed:
        return service.price
    raise NotImplemented
