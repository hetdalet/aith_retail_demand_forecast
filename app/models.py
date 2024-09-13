from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from app.core.db import Base


class Task(Base):
    __tablename__ = "task"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[UUID]
    ml_service: Mapped[str] = mapped_column(ForeignKey("ml_service.name"))
    start: Mapped[datetime]
    end: Mapped[datetime | None]
    input: Mapped[str | None]
    output: Mapped[str | None]

    service: Mapped["MLService"] = relationship()


class MLService(Base):
    __tablename__ = "ml_service"

    name: Mapped[str] = mapped_column(primary_key=True)
    description: Mapped[str | None]
