from fastapi import APIRouter

from app.api.config import settings
from app.api.schemas.response import HealthResponse, ReadyResponse
from app.api.services.model_service import model_service


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    return HealthResponse(status="ok", service="api", version=settings.app_version)


@router.get("/ready", response_model=ReadyResponse)
async def health_ready() -> ReadyResponse:
    ready, artifacts, details = model_service.readiness()
    return ReadyResponse(
        status="ready" if ready else "not_ready",
        loaded_artifacts=artifacts,
        details=details,
    )
