# storage/object_store.py — Helios MinIO S3-compatible object store
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import io
import logging
from datetime import timedelta
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from config import cfg

logger = logging.getLogger("helios.storage.object_store")

_client: Minio | None = None


def _get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            cfg.minio_endpoint,
            access_key=cfg.minio_access_key,
            secret_key=cfg.minio_secret_key,
            secure=cfg.minio_secure,
        )
    return _client


def ensure_bucket() -> None:
    """Create the bucket if it doesn't exist (idempotent)."""
    client = _get_client()
    if not client.bucket_exists(cfg.minio_bucket):
        client.make_bucket(cfg.minio_bucket)
        logger.info("MinIO bucket created: %s", cfg.minio_bucket)


def upload(key: str, data: bytes | BinaryIO, content_type: str = "application/octet-stream") -> str:
    """Upload bytes or file-like object; return the object key."""
    client = _get_client()
    if isinstance(data, bytes):
        stream = io.BytesIO(data)
        length = len(data)
    else:
        stream = data
        length = -1  # unknown length — streaming upload

    client.put_object(
        cfg.minio_bucket, key, stream, length,
        content_type=content_type,
    )
    logger.info("Uploaded %s (%s bytes) to MinIO", key, length if length >= 0 else "?")
    return key


def download(key: str) -> bytes:
    """Download object content as bytes."""
    client = _get_client()
    response = client.get_object(cfg.minio_bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def delete(key: str) -> None:
    _get_client().remove_object(cfg.minio_bucket, key)
    logger.info("Deleted %s from MinIO", key)


def presigned_url(key: str, expires_hours: int = 1) -> str:
    """Generate a time-limited presigned GET URL."""
    url = _get_client().presigned_get_object(
        cfg.minio_bucket,
        key,
        expires=timedelta(hours=expires_hours),
    )
    return url


def list_keys(prefix: str = "") -> list[str]:
    """List all object keys under prefix."""
    objects = _get_client().list_objects(cfg.minio_bucket, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]


def copy(src_key: str, dst_key: str) -> None:
    """Server-side copy within the same bucket."""
    from minio.commonconfig import CopySource
    _get_client().copy_object(
        cfg.minio_bucket,
        dst_key,
        CopySource(cfg.minio_bucket, src_key),
    )
    logger.info("Copied %s → %s", src_key, dst_key)


def ping() -> bool:
    try:
        _get_client().bucket_exists(cfg.minio_bucket)
        return True
    except Exception as exc:
        logger.error("MinIO ping failed: %s", exc)
        return False
