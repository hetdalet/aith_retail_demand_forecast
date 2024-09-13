import pika
import redis
from fastapi import Request
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.core.settings import settings


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
