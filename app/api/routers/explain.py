from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.schemas.request import ExplainRequest
from app.api.schemas.response import ExplainResponse
from app.api.services.explanation_service import explanation_service


router = APIRouter(prefix="/v1", tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
async def explain(payload: ExplainRequest) -> ExplainResponse:
    try:
        result = await run_in_threadpool(
            explanation_service.explain,
            payload.question,
            payload.session_id,
            payload.prediction_context,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Explanation request timed out.") from exc
    return ExplainResponse(**result)
