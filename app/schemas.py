from typing import Optional
from uuid import UUID

from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel
from pydantic import Field
from pydantic import EmailStr

from app.domain_types import TransactionType
from app.domain_types import ServicePricingType


class UserBase(BaseModel):
    email: EmailStr

    class Config:
        from_attributes = True


class UserCreate(UserBase):
    id: Optional[int] = None
    password: str
    tg_id: Optional[int] = None

    class Config:
        from_attributes = True


class User(UserBase):
    id: int
    tg_id: Optional[int] = None
    tg_link_token: Optional[str] = None
    deactivated: Optional[bool] = None

    class Config:
        from_attributes = True


class UserExt(User):
    account: Optional["Account"] = None

    class Config:
        from_attributes = True


class AccountCreate(BaseModel):
    user_id: int
    balance: Optional[Decimal] = 0

    class Config:
        from_attributes = True


class Account(AccountCreate):
    id: int

    class Config:
        from_attributes = True


class TransactionCreate(BaseModel):
    type: TransactionType
    amount: Decimal = Field(ge=0)

    class Config:
        from_attributes = True


class TransactionInsert(TransactionCreate):
    user_id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class Transaction(TransactionCreate):
    id: int
    user: "User"
    timestamp: datetime

    class Config:
        from_attributes = True


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
    service: "Service"
    user: "User"

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


class Service(BaseModel):
    name: str
    description: str | None
    pricing_type: ServicePricingType
    price: Decimal

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: EmailStr | None = None


class HealthCheck(BaseModel):
    status: str


class TgLink(BaseModel):
    link: str


class TgAccLinkRequest(BaseModel):
    link_token: str
    tg_user_id: int

