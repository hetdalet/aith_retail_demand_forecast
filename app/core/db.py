from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.settings import settings


def _get_engine():
    return create_engine(settings.db_url, echo=True)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=_get_engine()
)
Base = declarative_base()
