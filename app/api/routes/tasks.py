from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from sqlalchemy.orm import Session

from app.schemas import Task
from app.schemas import TaskBase
from app.services import task as task_service
from ...core import deps

router = APIRouter()


@router.patch("/{key}")
def finish_task(key: UUID,
                task: TaskBase,
                db: Session = Depends(deps.get_db)):
    task_service.finish_task(db=db, key=key, output=task.output)


@router.get("/{key}", response_model=Task)
def get_task_by_key(key: UUID,
                    db: Session = Depends(deps.get_db)):
    task = task_service.get_by_key(db, key)
    return task
