from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.core.db import Base
from app.domain_types import TransactionType
from app.domain_types import ServicePricingType


class Task(Base):
    __tablename__ = "task"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[UUID]
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
    service_name: Mapped[str] = mapped_column(ForeignKey("service.name"))
    transaction_id: Mapped[int] = mapped_column(ForeignKey("transaction.id"))
    start: Mapped[datetime]
    end: Mapped[datetime | None]
    input: Mapped[str | None]
    output: Mapped[str | None]

    service: Mapped["Service"] = relationship()


class Service(Base):
    __tablename__ = "service"

    name: Mapped[str] = mapped_column(primary_key=True)
    description: Mapped[str | None]
    pricing_type: Mapped[ServicePricingType]
    price: Mapped[Decimal]
