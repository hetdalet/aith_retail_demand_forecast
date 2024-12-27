import os
from pydantic import PostgresDsn
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        secrets_dir=os.environ.get("SECRETS_DIR")
    )
    postgres_db: str = Field(alias="postgres_db")
    postgres_user: str = Field(alias="postgres_user")
    postgres_password: str = Field(alias="postgres_password")
    db_host: str
    db_port: int
    rabbitmq_user: str
    rabbitmq_password: str
    rabbitmq_host: str
    rabbitmq_port: int
    redis_host: str
    redis_port: int
    app_host: str
    app_port: int
    cookie_name: str

    @property
    def db_url(self) -> PostgresDsn:
        pg_url = "postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
            user=self.postgres_user,
            password=self.postgres_password,
            host=self.db_host,
            port=self.db_port,
            dbname=self.postgres_db,
        )
        return pg_url


settings = Settings()
