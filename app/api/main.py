from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.config import settings
from app.api.core.logging import configure_logging
from app.api.routers.health import router as health_router
from app.api.routers.predict import router as predict_router
from app.api.routers.explain import router as explain_router


configure_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Async API for diabetic readmission prediction and explanation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response.headers["x-trace-id"] = trace_id
    response.headers["x-latency-ms"] = str(elapsed_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    trace_id = request.headers.get("x-trace-id")
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "internal_error",
            "message": "Unexpected server error",
            "details": {"type": exc.__class__.__name__},
            "trace_id": trace_id,
        },
    )


app.include_router(health_router)
app.include_router(predict_router)
app.include_router(explain_router)
