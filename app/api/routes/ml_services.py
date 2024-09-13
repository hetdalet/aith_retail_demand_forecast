from typing import List

from fastapi import APIRouter
from fastapi import Depends
from pika.connection import Connection as RMQConn
from sqlalchemy.orm import Session

from app.schemas import Task
from app.schemas import TaskBase
from app.schemas import TaskStart
from app.services import task as task_service
from ...core import deps

router = APIRouter()


@router.post("/{service_name}/task", response_model=TaskStart)
def start_task(task: TaskBase,
               service_name: str,
               db: Session = Depends(deps.get_db),
               rmq: RMQConn = Depends(deps.get_rmq)):
    task = task_service.create_task(
        db=db,
        task=task,
        service_name=service_name
    )
    task_service.send_task_to_queue(rmq, task)
    return task


@router.get("/{service_name}/tasks", response_model=List[Task])
def list_tasks(service_name: str,
               limit: int = None,
               offset: int = None,
               db: Session = Depends(deps.get_db)):
    options = [("ml_service", "==", service_name)]
    tasks = task_service.list_by_options(
        db=db,
        options=options,
        limit=limit,
        offset=offset
    )
    return tasks
