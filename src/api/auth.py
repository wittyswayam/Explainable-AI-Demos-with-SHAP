"""
src/api/auth.py
================
JWT + API Key authentication for the XAI Platform API.

Supports two auth modes:
1. Bearer JWT tokens (issued by /auth/token endpoint)
2. Static API keys (X-API-Key header, stored in Redis or env)

Usage in routers:
    from src.api.auth import require_auth
    @router.get("/protected")
    async def protected(user: TokenData = Depends(require_auth)):
        ...
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from jose import JWTError, jwt
from pydantic import BaseModel

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Token data
# ---------------------------------------------------------------------------

class TokenData(BaseModel):
    sub: str                      # subject (user_id or service name)
    exp: Optional[datetime] = None
    scopes: list[str] = []
    is_api_key: bool = False


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int               # seconds


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def create_access_token(
    subject: str,
    scopes: list[str] | None = None,
    expires_minutes: int | None = None,
) -> Token:
    """Issue a signed JWT access token."""
    expire_minutes = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "scopes": scopes or [],
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return Token(access_token=token, expires_in=expire_minutes * 60)


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return TokenData(
            sub=payload["sub"],
            exp=payload.get("exp"),
            scopes=payload.get("scopes", []),
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _validate_api_key(api_key: str) -> TokenData:
    """Validate a static API key against the configured master key."""
    master_key = os.environ.get("MASTER_API_KEY", "")
    if not master_key or api_key != master_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return TokenData(sub="api-key-client", scopes=["read", "explain"], is_api_key=True)


# ---------------------------------------------------------------------------
# Dependency: require_auth
# ---------------------------------------------------------------------------

async def require_auth(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_scheme),
) -> TokenData:
    """
    FastAPI dependency: validates JWT Bearer token or X-API-Key header.
    Raises HTTP 401 if neither is present or valid.
    """
    if bearer and bearer.credentials:
        return decode_token(bearer.credentials)

    if api_key:
        return _validate_api_key(api_key)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def optional_auth(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_scheme),
) -> Optional[TokenData]:
    """
    Optional auth dependency — returns None if no credentials supplied.
    Use for endpoints that have different behaviour for authenticated users.
    """
    try:
        return await require_auth(bearer, api_key)
    except HTTPException:
        return None


def require_scope(scope: str):
    """Factory: return dependency that requires a specific scope."""
    async def _check_scope(user: TokenData = Depends(require_auth)) -> TokenData:
        if scope not in user.scopes and not user.is_api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Requires scope: '{scope}'",
            )
        return user
    return _check_scope


# ---------------------------------------------------------------------------
# Auth router (token endpoint)
# ---------------------------------------------------------------------------

from fastapi import APIRouter, Form

auth_router = APIRouter(prefix="/auth", tags=["Auth"])


@auth_router.post("/token", response_model=Token)
async def issue_token(
    username: str = Form(...),
    password: str = Form(...),
) -> Token:
    """
    Issue a JWT access token (password flow).

    In production, verify credentials against user store (database/LDAP).
    This implementation uses environment-variable credentials for simplicity.
    """
    valid_user = os.environ.get("API_USERNAME", "admin")
    valid_pass = os.environ.get("API_PASSWORD", "")

    if not valid_pass:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth not configured: set API_USERNAME and API_PASSWORD env vars",
        )

    if username != valid_user or password != valid_pass:
        logger.warning("Failed login attempt for user: %s", username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    token = create_access_token(subject=username, scopes=["read", "explain", "admin"])
    logger.info("Token issued for user: %s", username)
    return token
