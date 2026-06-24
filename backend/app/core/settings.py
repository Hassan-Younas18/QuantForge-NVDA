"""Backend settings — CORS origins and defaults, overridable via env vars."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NVDA_API_", extra="ignore")

    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    default_ticker: str = "NVDA"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
