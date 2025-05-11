from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    config_file_path: Path = Path("config.json")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
