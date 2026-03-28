from datetime import datetime, UTC
import random

from fastapi import APIRouter

from app.api.schemas.request import DummyRequest
from app.api.schemas.response import DummyResponse


router = APIRouter(prefix="/v1", tags=["utility"])


@router.post("/dummy", response_model=DummyResponse)
async def dummy_value(payload: DummyRequest) -> DummyResponse:
    rng = random.Random(payload.seed)
    return DummyResponse(
        generated_value=round(rng.random(), 6),
        generated_at=datetime.now(UTC).isoformat(),
    )
