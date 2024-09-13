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
from app.models import Task as TaskDB
from app.schemas import Task
from app.schemas import TaskBase
from app.schemas import TaskInsert
from app.schemas import TaskStart
from app.schemas import TaskFinish
from app import repository

from .ml_service import get_service


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
                service_name: str) -> TaskStart:
    task = repository.create(
        TaskDB,
        item=TaskInsert(
            key=uuid4(),
            input=task.input,
            start=datetime.now(),
            ml_service_name=service_name,
        ),
        result_schema=TaskStart,
        db=db
    )
    return task


def send_task_to_queue(rmq: RMQConn, task: TaskDB):
    channel = rmq.channel()
    ml_service_name = task.ml_service.name
    channel.queue_declare(queue=ml_service_name, durable=True)
    app_host = settings.app_host
    app_port = settings.app_port
    message = pickle.dumps({
        "key": task.key,
        "input": task.input,
        "callback_ep": f"http://{app_host}:{app_port}/tasks/{task.key}"
    })
    channel.basic_publish(
        exchange='',
        routing_key=ml_service_name,
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
