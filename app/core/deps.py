import pika
import redis
from fastapi import Request
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from sqlalchemy.orm import Session

from app.core import auth
from app.core.db import SessionLocal
from app.core.settings import settings
from app.services import user as user_service


def _get_session_local():
    return SessionLocal()


def get_db(db: Session = Depends(_get_session_local)):
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    else:
        db.commit()
    finally:
        db.close()


def get_rmq():
    conn = pika.BlockingConnection(
        pika.ConnectionParameters(
            host='rabbitmq',
            credentials=pika.PlainCredentials(
                username=settings.rabbitmq_user,
                password=settings.rabbitmq_password,
            )
        )
    )
    try:
        yield conn
    finally:
        conn.close()


def get_rds():
    rds = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True
    )
    try:
        yield rds
    finally:
        rds.close()


def get_cur_user_from_token(token: str = Depends(auth.oauth2_scheme),
                            db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token_data = auth.decode_token(token)
    except Exception:
        raise credentials_exception
    user = user_service.get_by_email(db, email=token_data.username)
    if user is None or user.deactivated is True:
        raise credentials_exception
    return user


def get_cur_user_from_cookie(request: Request,
                             db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token = request.cookies.get(settings.cookie_name)
    user = auth.authenticate_cookie(db, token)
    if user is None or user.deactivated is True:
        raise credentials_exception
    return user
