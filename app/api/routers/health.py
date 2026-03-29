from fastapi import APIRouter

from app.api.config import settings
from app.api.schemas.response import HealthResponse, ReadyResponse
from app.api.services.model_service import model_service


router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/live",
    response_model=HealthResponse,
    summary="Liveness Check",
    description="Lightweight health endpoint used by orchestrators and uptime checks.",
    response_description="Service liveness payload.",
)
async def health_live() -> HealthResponse:
    return HealthResponse(status="ok", service="api", version=settings.app_version)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness Check",
    description="Readiness endpoint that verifies model artifacts are available and loadable.",
    response_description="Readiness status and loaded artifact details.",
)
async def health_ready() -> ReadyResponse:
    ready, artifacts, details = model_service.readiness()
    return ReadyResponse(
        status="ready" if ready else "not_ready",
        loaded_artifacts=artifacts,
        details=details,
    )
