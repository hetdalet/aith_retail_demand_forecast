import pickle
from datetime import datetime
from typing import Tuple
from typing import Sequence
from typing import Any
from uuid import UUID
from uuid import uuid4

import pika
from pika.connection import Connection as RMQConn
from sqlalchemy.orm import Session

from app.core.settings import settings
from app.domain_types import TransactionType
from app.models import Task as TaskDB
from app.schemas import Task
from app.schemas import TaskBase
from app.schemas import TaskInsert
from app.schemas import TaskStart
from app.schemas import TaskFinish
from app.schemas import TransactionCreate
from app.schemas import User
from app import repository

from .service import get_service
from .service import calc_service_price
from .transaction import create_transaction


def get_taskr(task_id: int, db: Session):
    return repository.read_by_id(TaskDB, task_id, Task, db)


def get_by_key(db: Session, key: UUID):
    return repository.read_by_id(TaskDB, key, Task, db, id_field="key")


def list_by_options(db: Session,
                    options: Sequence[Tuple[str, str, Any]] = None,
                    order: Tuple[str, str] = ("start", "asc"),
                    limit: int = None,
                    offset: int = None) -> Sequence[Task]:
    tasks = repository.list_by_options(
        TaskDB,
        result_schema=Task,
        options=options,
        order=order,
        limit=limit,
        offset=offset,
        db=db
    )
    return tasks


def create_task(db: Session,
                task: TaskBase,
                user: User,
                service_name: str) -> TaskStart:
    service = get_service(service_name, db)
    price = calc_service_price(service, task.input)
    transaction = TransactionCreate(type=TransactionType.wdr, amount=price)
    transaction = create_transaction(db, transaction, user)
    task = repository.create(
        TaskDB,
        item=TaskInsert(
            key=uuid4(),
            input=task.input,
            start=datetime.now(),
            user_id=user.id,
            service_name=service_name,
            transaction_id=transaction.id
        ),
        result_schema=TaskStart,
        db=db
    )
    return task


def send_task_to_queue(rmq: RMQConn, task: TaskDB):
    channel = rmq.channel()
    service_name = task.service.name
    channel.queue_declare(queue=service_name, durable=True)
    app_host = settings.app_host
    app_port = settings.app_port
    message = pickle.dumps({
        "key": task.key,
        "input": task.input,
        "callback_ep": f"http://{app_host}:{app_port}/tasks/{task.key}"
    })
    channel.basic_publish(
        exchange='',
        routing_key=service_name,
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=pika.DeliveryMode.Persistent
        )
    )


def finish_task(db: Session, key: UUID, output: str):
    task = repository.update(
        TaskDB,
        item=TaskFinish(
            key=key,
            end=datetime.now(),
            output=output
        ),
        id_field="key",
        result_schema=Task,
        db=db,
    )
    return task
