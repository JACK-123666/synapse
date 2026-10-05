"""安全工具：密码哈希、JWT、API Key、敏感字段加密。"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import bcrypt
import jwt
from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings

logger = logging.getLogger(__name__)

_JWT_ALGORITHM = "HS256"
_API_KEY_PREFIX = "syn-"

_secret_lock = threading.Lock()
_cached_secret: Optional[str] = None


def get_app_secret() -> str:
    """应用密钥：优先 APP_SECRET_KEY；否则读取 / 生成 data_dir/.secret_key。

    密钥持久化到文件，保证重启后 JWT 和加密数据依然有效。
    """
    global _cached_secret
    if _cached_secret:
        return _cached_secret
    with _secret_lock:
        if _cached_secret:
            return _cached_secret
        settings = get_settings()
        if settings.app_secret_key:
            _cached_secret = settings.app_secret_key
            return _cached_secret
        path = settings.data_path(".secret_key")
        if path.exists():
            _cached_secret = path.read_text(encoding="utf-8").strip()
        if not _cached_secret:
            _cached_secret = secrets.token_urlsafe(48)
            path.write_text(_cached_secret, encoding="utf-8")
            logger.warning("未配置 APP_SECRET_KEY，已生成并保存到 %s", path)
        return _cached_secret


def reset_secret_cache() -> None:
    global _cached_secret
    _cached_secret = None


# ---- 密码 ----


def _pw_bytes(password: str) -> bytes:
    # bcrypt 只使用前 72 字节，超出部分显式截断
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_pw_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(_pw_bytes(password), hashed.encode("utf-8"))
    except ValueError:
        return False


# ---- JWT ----


def create_access_token(user_id: str, username: str, role: str) -> Tuple[str, int]:
    """签发访问令牌，返回 (token, 有效秒数)。"""
    expire_minutes = get_settings().jwt_expire_minutes
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "name": username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expire_minutes)).timestamp()),
    }
    token = jwt.encode(payload, get_app_secret(), algorithm=_JWT_ALGORITHM)
    return token, expire_minutes * 60


def decode_access_token(token: str) -> Dict[str, Any]:
    """校验并解析访问令牌；无效或过期时抛出 jwt.PyJWTError。"""
    return jwt.decode(token, get_app_secret(), algorithms=[_JWT_ALGORITHM])


# ---- API Key ----


def hash_api_key(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def generate_api_key() -> Tuple[str, str, str]:
    """生成 API Key，返回 (明文, 展示前缀, 哈希)。"""
    plain = _API_KEY_PREFIX + secrets.token_urlsafe(32)
    return plain, plain[:12], hash_api_key(plain)


def looks_like_api_key(value: str) -> bool:
    return value.startswith(_API_KEY_PREFIX)


# ---- 敏感字段加密 ----


def _fernet() -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(get_app_secret().encode("utf-8")).digest())
    return Fernet(key)


def encrypt_secret(plain: str) -> str:
    if not plain:
        return ""
    return _fernet().encrypt(plain.encode("utf-8")).decode("utf-8")


def decrypt_secret(token: str) -> str:
    if not token:
        return ""
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.error("敏感字段解密失败：应用密钥可能已变更")
        return ""
