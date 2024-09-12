import os
import uvicorn
from alembic.config import Config
from alembic import command


if __name__ == "__main__":
    alembic_cfg = Config(os.environ["ALEMBIC_CONFIG"])
    alembic_cfg.set_main_option("script_location", "app:migrations")
    command.upgrade(alembic_cfg, "head")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("APP_PORT", 8000)),
        reload=True,
        reload_dirs=["/app/"],
    )
