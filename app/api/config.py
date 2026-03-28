from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    app_name: str = "Diabetic Readmission API"
    app_version: str = "0.1.0"
    app_env: str = "dev"
    log_level: str = "INFO"

    model_dir: Path = Path("models")
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-nano"

    cors_allow_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
