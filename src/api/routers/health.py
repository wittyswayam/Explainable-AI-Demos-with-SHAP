"""
API Routers — Health
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="2.0.0", service="xai-platform")


@router.get("/live")
async def liveness() -> dict:
    return {"alive": True}


@router.get("/ready")
async def readiness() -> dict:
    return {"ready": True}
