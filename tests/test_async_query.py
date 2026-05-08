# tests/test_async_query.py — Tests for POST /query/async and GET /query/task/{task_id}
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture()
def client():
    with (
        patch("storage.database.get_engine"),
        patch("storage.database.create_tables", new_callable=AsyncMock),
        patch("storage.object_store.ensure_bucket"),
        patch("observability.tracing.setup_tracing"),
        patch("observability.logging_config.setup_logging"),
        patch("storage.database.close_engine", new_callable=AsyncMock),
        patch("storage.read_replica.close_read_engine", new_callable=AsyncMock),
        patch("storage.cache.incr", new_callable=AsyncMock, return_value=1),
        patch("storage.cache.ttl_seconds", new_callable=AsyncMock, return_value=60),
    ):
        from main import app
        return TestClient(app)


def _make_token(user_id: str = "u1") -> str:
    from jose import jwt
    from config import cfg
    from datetime import datetime, timedelta, timezone
    return jwt.encode(
        {"sub": user_id, "type": "access",
         "exp": datetime.now(timezone.utc) + timedelta(minutes=60)},
        cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm,
    )


class TestQueryAsync:

    def test_async_query_returns_202_with_task_and_query_ids(self, client):
        mock_user = MagicMock(id="u1", is_active=True)
        mock_task = MagicMock(id="celery-task-abc123")

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.add = MagicMock()

        with (
            patch("api.auth.get_user_by_id", new_callable=AsyncMock, return_value=mock_user),
            patch("storage.database.get_session", return_value=mock_session),
            patch("api.routes.run_pipeline_task") as mock_task_fn,
            patch("resilience.backpressure.check_backpressure", new_callable=AsyncMock),
        ):
            mock_task_fn.delay.return_value = mock_task
            resp = client.post(
                "/api/v1/query/async",
                json={"query": "What is the capital of France?"},
                headers={"Authorization": f"Bearer {_make_token()}"},
            )

        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert "query_id" in body
        assert body["status"] == "queued"
        assert body["task_id"] == "celery-task-abc123"

    def test_async_query_creates_queued_record(self, client):
        """QueryRecord with status='queued' must be added before task dispatch."""
        mock_user = MagicMock(id="u1", is_active=True)
        mock_task = MagicMock(id="task-id-xyz")
        added_records = []

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        def capture_add(record):
            added_records.append(record)

        mock_session.add = capture_add

        with (
            patch("api.auth.get_user_by_id", new_callable=AsyncMock, return_value=mock_user),
            patch("storage.database.get_session", return_value=mock_session),
            patch("api.routes.run_pipeline_task") as mock_task_fn,
            patch("resilience.backpressure.check_backpressure", new_callable=AsyncMock),
        ):
            mock_task_fn.delay.return_value = mock_task
            resp = client.post(
                "/api/v1/query/async",
                json={"query": "test query"},
                headers={"Authorization": f"Bearer {_make_token()}"},
            )

        assert resp.status_code == 202
        assert len(added_records) == 1
        record = added_records[0]
        assert record.status == "queued"
        assert record.user_id == "u1"
        assert record.query_text == "test query"

    def test_async_query_requires_auth(self, client):
        resp = client.post("/api/v1/query/async", json={"query": "test"})
        assert resp.status_code == 401

    def test_async_query_passes_query_id_to_task(self, client):
        """The task must receive query_id so it can write results back."""
        mock_user = MagicMock(id="u1", is_active=True)
        mock_task = MagicMock(id="task-qid-test")
        call_kwargs = {}

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.add = MagicMock()

        with (
            patch("api.auth.get_user_by_id", new_callable=AsyncMock, return_value=mock_user),
            patch("storage.database.get_session", return_value=mock_session),
            patch("api.routes.run_pipeline_task") as mock_task_fn,
            patch("resilience.backpressure.check_backpressure", new_callable=AsyncMock),
        ):
            def capture_delay(*args, **kwargs):
                call_kwargs.update(kwargs)
                return mock_task

            mock_task_fn.delay = capture_delay
            resp = client.post(
                "/api/v1/query/async",
                json={"query": "trace this"},
                headers={"Authorization": f"Bearer {_make_token()}"},
            )

        assert resp.status_code == 202
        query_id = resp.json()["query_id"]
        assert call_kwargs.get("query_id") == query_id


class TestTaskStatus:

    def test_task_status_pending_returns_queued(self, client):
        mock_user = MagicMock(id="u1", is_active=True)

        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_result.result = None
        mock_result.info = None

        with (
            patch("api.auth.get_user_by_id", new_callable=AsyncMock, return_value=mock_user),
            patch("api.routes.AsyncResult", return_value=mock_result),
        ):
            resp = client.get(
                "/api/v1/query/task/some-task-id",
                headers={"Authorization": f"Bearer {_make_token()}"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["celery_state"] == "PENDING"
        assert body["status"] == "queued"
        assert body["result"] is None

    def test_task_status_success_returns_done_with_result(self, client):
        mock_user = MagicMock(id="u1", is_active=True)

        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"answer": "Paris", "critic_passed": True, "critic_scores": None}
        mock_result.info = {"query_id": "qid-123"}

        with (
            patch("api.auth.get_user_by_id", new_callable=AsyncMock, return_value=mock_user),
            patch("api.routes.AsyncResult", return_value=mock_result),
        ):
            resp = client.get(
                "/api/v1/query/task/done-task-id",
                headers={"Authorization": f"Bearer {_make_token()}"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["celery_state"] == "SUCCESS"
        assert body["status"] == "done"
        assert body["result"]["answer"] == "Paris"

    def test_task_status_failure_returns_failed(self, client):
        mock_user = MagicMock(id="u1", is_active=True)

        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.result = None
        mock_result.info = None

        with (
            patch("api.auth.get_user_by_id", new_callable=AsyncMock, return_value=mock_user),
            patch("api.routes.AsyncResult", return_value=mock_result),
        ):
            resp = client.get(
                "/api/v1/query/task/failed-task-id",
                headers={"Authorization": f"Bearer {_make_token()}"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"

    def test_task_status_requires_auth(self, client):
        resp = client.get("/api/v1/query/task/some-task-id")
        assert resp.status_code == 401
