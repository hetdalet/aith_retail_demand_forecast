import operator
from typing import Any, Sequence
from typing import Tuple

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.db import Base


OP_MAP = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def create(db_model: Base,
           item: BaseModel,
           result_schema: BaseModel,
           db: Session) -> Base:
    db_item = db_model(**item.dict())
    db.add(db_item)
    db.flush()
    db.refresh(db_item)
    return result_schema.model_validate(db_item)


def read_by_id(db_model: Base,
               item_id: Any,
               result_schema: BaseModel,
               db: Session,
               id_field: str = 'id'):
    item = (
        db.query(db_model)
        .filter(getattr(db_model, id_field) == item_id)
        .first()
    )
    if not item:
        return None
    return result_schema.model_validate(item)


def list_by_options(db_model: Base,
                    result_schema: BaseModel,
                    options: Sequence[Tuple[str, str, Any]],
                    order: Tuple[str, str],
                    limit: int,
                    offset: int,
                    db: Session) -> BaseModel:
    stmt = sa.select(db_model)

    for option in options:
        stmt = stmt.where(_option_to_where_clause(db_model, option))
    if order:
        stmt = stmt.order_by(_get_order_by_clause(db_model, *order))
    if limit:
        stmt = stmt.limit(limit)
    if offset:
        stmt = stmt.offset(offset)

    result = [
        result_schema.model_validate(row)
        for row in db.scalars(stmt).all()
    ]
    return result


def update(db_model: Base,
           item: BaseModel,
           result_schema: BaseModel,
           db: Session,
           id_field: str = "id"):
    stmt = (
        sa.update(db_model)
        .where(getattr(db_model, id_field) == getattr(item, id_field))
        .values(**item.dict())
        .returning(db_model)
    )
    updated_item = db.scalars(stmt).first()
    db.flush()
    return result_schema.model_validate(updated_item)


def _option_to_where_clause(db_model: Base, option: tuple):
    field, op_str, value = option
    op = OP_MAP[op_str]
    return op(getattr(db_model, field), value)


def _get_order_by_clause(model, field, direction):
    return getattr(getattr(model, field), direction)()
