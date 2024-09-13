from typing import Optional
from uuid import UUID

from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel
from pydantic import Field
from pydantic import EmailStr


class TaskBase(BaseModel):
    input: Optional[str] = None
    output: Optional[str] = None

    class Config:
        from_attributes = True


class TaskInsert(TaskBase):
    key: UUID
    start: datetime
    user_id: int
    service_name: str
    transaction_id: int

    class Config:
        from_attributes = True


class TaskStart(TaskBase):
    id: int
    key: UUID
    start: datetime
    ml_service: "MLService"

    class Config:
        from_attributes = True


class TaskFinish(BaseModel):
    key: UUID
    output: str | None
    end: datetime

    class Config:
        from_attributes = True


class Task(TaskStart):
    end: datetime | None

    class Config:
        from_attributes = True


class MLService(BaseModel):
    name: str
    description: str | None

    class Config:
        from_attributes = True


class HealthCheck(BaseModel):
    status: str
