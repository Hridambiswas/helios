# tests/test_storage.py — Unit tests for storage layer
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch


class TestCache:
    """Tests for storage/cache.py — no live Redis needed."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        from storage import cache
        mock_client = AsyncMock()
        mock_client.setex = AsyncMock()
        mock_client.get = AsyncMock(return_value=json.dumps({"key": "value"}))

        with patch("storage.cache._client", return_value=mock_client):
            await cache.set("test", "k1", {"key": "value"})
            result = await cache.get("test", "k1")

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        from storage import cache
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)

        with patch("storage.cache._client", return_value=mock_client):
            result = await cache.get("test", "missing_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_incr_returns_count(self):
        from storage import cache
        mock_client = AsyncMock()
        mock_client.incr = AsyncMock(return_value=1)
        mock_client.expire = AsyncMock()

        with patch("storage.cache._client", return_value=mock_client):
            count = await cache.incr("ratelimit", "1.2.3.4")

        assert count == 1


class TestObjectStore:
    """Tests for storage/object_store.py."""

    def test_upload_calls_put_object(self):
        from storage import object_store
        mock_client = MagicMock()

        with patch("storage.object_store._get_client", return_value=mock_client):
            object_store.upload("test/key.txt", b"hello world")

        mock_client.put_object.assert_called_once()

    def test_presigned_url_returns_string(self):
        from storage import object_store
        mock_client = MagicMock()
        mock_client.presigned_get_object.return_value = "https://minio/signed/url"

        with patch("storage.object_store._get_client", return_value=mock_client):
            url = object_store.presigned_url("test/key.txt", expires_hours=2)

        assert url == "https://minio/signed/url"

    def test_ping_returns_true_on_success(self):
        from storage import object_store
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True

        with patch("storage.object_store._get_client", return_value=mock_client):
            result = object_store.ping()

        assert result is True
