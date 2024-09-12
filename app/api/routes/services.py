from fastapi import APIRouter
from fastapi import Depends
from pika.connection import Connection as RMQConn
from sqlalchemy.orm import Session

from app.schemas import TaskBase
from app.schemas import TaskStart
from app.schemas import User
from app.services import task as task_service
from ...core import deps

router = APIRouter()


@router.post("/{service_name}/task", response_model=TaskStart)
def start_task(task: TaskBase,
               service_name: str,
               db: Session = Depends(deps.get_db),
               rmq: RMQConn = Depends(deps.get_rmq),
               user: User = Depends(deps.get_cur_user_from_cookie)):
    task = task_service.create_task(
        db=db,
        task=task,
        user=user,
        service_name=service_name
    )
    task_service.send_task_to_queue(rmq, task)
    return task
