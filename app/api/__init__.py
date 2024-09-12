from fastapi import APIRouter

from .routes import auth
from .routes import index
from .routes import health
from .routes import token
from .routes import services
from .routes import tasks
from .routes import transactions
from .routes import users


api_router = APIRouter()
api_router.include_router(index.router)
api_router.include_router(health.router, prefix="/health")
api_router.include_router(auth.router, prefix="/auth")
api_router.include_router(token.router, prefix="/token")
api_router.include_router(services.router, prefix="/services")
api_router.include_router(tasks.router, prefix="/tasks")
api_router.include_router(transactions.router, prefix="/transactions")
api_router.include_router(users.router, prefix="/users")
