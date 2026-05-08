# tests/test_resilience.py — Unit tests for Helios resilience patterns
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import asyncio
import threading
import time
import pytest
from unittest.mock import patch, MagicMock


# ── Circuit Breaker ───────────────────────────────────────────────────────────

class TestCircuitBreaker:

    def _make_breaker(self, threshold=3, recovery=30.0):
        from resilience.circuit_breaker import CircuitBreaker
        return CircuitBreaker("test", failure_threshold=threshold, recovery_timeout=recovery)

    def test_closed_by_default(self):
        from resilience.circuit_breaker import CircuitState
        cb = self._make_breaker()
        assert cb.state == CircuitState.CLOSED

    def test_successful_call_keeps_closed(self):
        from resilience.circuit_breaker import CircuitState
        cb = self._make_breaker()
        result = cb.call_sync(lambda: 42)
        assert result == 42
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_trips_to_open_after_threshold(self):
        from resilience.circuit_breaker import CircuitState
        cb = self._make_breaker(threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call_sync(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert cb.state == CircuitState.OPEN

    def test_open_raises_circuit_breaker_open(self):
        from resilience.circuit_breaker import CircuitState, CircuitBreakerOpen
        cb = self._make_breaker(threshold=1)
        with pytest.raises(RuntimeError):
            cb.call_sync(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpen):
            cb.call_sync(lambda: "should not run")

    def test_transitions_to_half_open_after_timeout(self):
        from resilience.circuit_breaker import CircuitState
        cb = self._make_breaker(threshold=1, recovery=0.01)
        with pytest.raises(RuntimeError):
            cb.call_sync(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        time.sleep(0.05)
        # Probe: success in HALF_OPEN → CLOSED
        result = cb.call_sync(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_returns_to_open(self):
        from resilience.circuit_breaker import CircuitState
        cb = self._make_breaker(threshold=1, recovery=0.01)
        with pytest.raises(RuntimeError):
            cb.call_sync(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        time.sleep(0.05)
        # Probe fails: should go back to OPEN
        with pytest.raises(RuntimeError):
            cb.call_sync(lambda: (_ for _ in ()).throw(RuntimeError("still broken")))
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = self._make_breaker(threshold=5)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call_sync(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        cb.call_sync(lambda: None)
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_async_call_wrapper(self):
        from resilience.circuit_breaker import CircuitState
        cb = self._make_breaker()

        async def good_coro():
            return "async-ok"

        result = await cb.call_async(good_coro)
        assert result == "async-ok"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_failure_increments_count(self):
        cb = self._make_breaker(threshold=3)

        async def bad_coro():
            raise ValueError("async-fail")

        with pytest.raises(ValueError):
            await cb.call_async(bad_coro)
        assert cb.failure_count == 1

    def test_get_breaker_registry_returns_same_instance(self):
        from resilience.circuit_breaker import get_breaker, _registry
        b1 = get_breaker("registry-test-cb")
        b2 = get_breaker("registry-test-cb")
        assert b1 is b2
        _registry.pop("registry-test-cb", None)

    def test_thread_safety_concurrent_failures(self):
        from resilience.circuit_breaker import CircuitState
        cb = self._make_breaker(threshold=10)
        errors = []

        def fail_once():
            try:
                cb.call_sync(lambda: (_ for _ in ()).throw(RuntimeError("t")))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fail_once) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 10
        assert cb.state == CircuitState.OPEN


# ── Bulkhead ──────────────────────────────────────────────────────────────────

class TestBulkhead:

    def test_single_acquisition_succeeds(self):
        from resilience.bulkhead import bulkhead, _pools
        _pools.pop("test-bh", None)
        with patch("config.cfg.bulkhead_default_limit", 2):
            with bulkhead("test-bh"):
                pass  # should not raise

    def test_rejects_when_all_slots_occupied(self):
        from resilience.bulkhead import bulkhead, BulkheadRejected, _pools
        _pools.pop("test-full-bh", None)

        acquired = []
        with patch("config.cfg.bulkhead_default_limit", 1):
            bh1 = bulkhead("test-full-bh")
            bh1.__enter__()
            acquired.append(bh1)
            try:
                with pytest.raises(BulkheadRejected):
                    with bulkhead("test-full-bh"):
                        pass
            finally:
                bh1.__exit__(None, None, None)

    def test_slot_released_after_context_exit(self):
        from resilience.bulkhead import bulkhead, _pools
        _pools.pop("test-release-bh", None)
        with patch("config.cfg.bulkhead_default_limit", 1):
            with bulkhead("test-release-bh"):
                pass
            # Slot released — second acquisition must succeed
            with bulkhead("test-release-bh"):
                pass

    def test_slot_released_even_on_exception(self):
        from resilience.bulkhead import bulkhead, _pools
        _pools.pop("test-exc-bh", None)
        with patch("config.cfg.bulkhead_default_limit", 1):
            try:
                with bulkhead("test-exc-bh"):
                    raise ValueError("inside bulkhead")
            except ValueError:
                pass
            # Must still be acquirable
            with bulkhead("test-exc-bh"):
                pass

    def test_rejection_increments_metric(self):
        from resilience.bulkhead import bulkhead, BulkheadRejected, _pools
        _pools.pop("test-metric-bh", None)
        counter = MagicMock()
        counter.labels.return_value = counter
        with (
            patch("config.cfg.bulkhead_default_limit", 1),
            patch("observability.metrics.bulkhead_rejected_counter", counter),
        ):
            bh = bulkhead("test-metric-bh")
            bh.__enter__()
            try:
                with pytest.raises(BulkheadRejected):
                    with bulkhead("test-metric-bh"):
                        pass
                counter.labels.assert_called_once_with(agent="test-metric-bh")
                counter.inc.assert_called_once()
            finally:
                bh.__exit__(None, None, None)


# ── Backpressure ──────────────────────────────────────────────────────────────

class TestBackpressure:

    def test_allows_requests_below_pipeline_threshold(self):
        import asyncio
        from resilience import backpressure as bp
        bp._active_pipelines = 0
        with patch("config.cfg.backpressure_active_pipelines_threshold", 20):
            asyncio.get_event_loop().run_until_complete(bp.check_backpressure())  # must not raise

    def test_raises_when_pipeline_limit_reached(self):
        import asyncio
        from resilience import backpressure as bp
        from resilience.backpressure import BackpressureError
        bp._active_pipelines = 20
        with patch("config.cfg.backpressure_active_pipelines_threshold", 20):
            with pytest.raises(BackpressureError):
                asyncio.get_event_loop().run_until_complete(bp.check_backpressure())
        bp._active_pipelines = 0

    @pytest.mark.asyncio
    async def test_active_pipeline_counter_increments_and_decrements(self):
        from resilience import backpressure as bp
        from resilience.backpressure import active_pipeline
        initial = bp.get_active_pipeline_count()
        async with active_pipeline():
            assert bp.get_active_pipeline_count() == initial + 1
        assert bp.get_active_pipeline_count() == initial

    @pytest.mark.asyncio
    async def test_active_pipeline_decrements_on_exception(self):
        from resilience import backpressure as bp
        from resilience.backpressure import active_pipeline
        initial = bp.get_active_pipeline_count()
        try:
            async with active_pipeline():
                raise RuntimeError("mid-pipeline failure")
        except RuntimeError:
            pass
        assert bp.get_active_pipeline_count() == initial

    @pytest.mark.asyncio
    async def test_celery_queue_depth_sheds_load(self):
        from resilience.backpressure import check_backpressure, BackpressureError
        from resilience import backpressure as bp
        bp._active_pipelines = 0

        mock_client = MagicMock()
        mock_client.llen = AsyncMock(return_value=200)
        mock_client.aclose = AsyncMock()

        with (
            patch("config.cfg.backpressure_active_pipelines_threshold", 100),
            patch("config.cfg.backpressure_queue_depth_threshold", 100),
            patch("redis.asyncio.from_url", return_value=mock_client),
        ):
            with pytest.raises(BackpressureError, match="Queue overloaded"):
                await check_backpressure()

    @pytest.mark.asyncio
    async def test_redis_failure_does_not_shed(self):
        from resilience.backpressure import check_backpressure
        from resilience import backpressure as bp
        bp._active_pipelines = 0
        with (
            patch("config.cfg.backpressure_active_pipelines_threshold", 100),
            patch("redis.asyncio.from_url", side_effect=ConnectionError("redis down")),
        ):
            await check_backpressure()  # must not raise


# ── Saga ──────────────────────────────────────────────────────────────────────

class TestSaga:

    @pytest.mark.asyncio
    async def test_happy_path_runs_all_steps(self):
        from resilience.saga import Saga
        log: list[str] = []
        saga = (
            Saga("test")
            .step("a", lambda: log.append("a"), lambda: log.append("undo-a"))
            .step("b", lambda: log.append("b"), lambda: log.append("undo-b"))
        )
        await saga.execute()
        assert log == ["a", "b"]

    @pytest.mark.asyncio
    async def test_failure_triggers_reverse_compensation(self):
        from resilience.saga import Saga, SagaExecutionError
        log: list[str] = []

        def fail_step():
            raise RuntimeError("boom")

        saga = (
            Saga("test")
            .step("a", lambda: log.append("a"), lambda: log.append("undo-a"))
            .step("b", fail_step, lambda: log.append("undo-b"))
        )
        with pytest.raises(SagaExecutionError) as exc_info:
            await saga.execute()

        assert "b" == exc_info.value.step
        assert "undo-a" in log
        assert exc_info.value.compensated is True

    @pytest.mark.asyncio
    async def test_compensation_runs_in_reverse_order(self):
        from resilience.saga import Saga, SagaExecutionError
        order: list[str] = []

        saga = (
            Saga("test")
            .step("1", lambda: order.append("do-1"), lambda: order.append("undo-1"))
            .step("2", lambda: order.append("do-2"), lambda: order.append("undo-2"))
            .step("3", lambda: (_ for _ in ()).throw(RuntimeError("fail")), lambda: None)
        )
        with pytest.raises(SagaExecutionError):
            await saga.execute()

        undo_order = [x for x in order if x.startswith("undo")]
        assert undo_order == ["undo-2", "undo-1"]

    @pytest.mark.asyncio
    async def test_async_action_is_awaited(self):
        from resilience.saga import Saga
        log: list[str] = []

        async def async_step():
            log.append("async-done")

        saga = Saga("test").step("a", async_step, lambda: None)
        await saga.execute()
        assert "async-done" in log

    @pytest.mark.asyncio
    async def test_async_compensator_is_awaited(self):
        from resilience.saga import Saga, SagaExecutionError
        log: list[str] = []

        async def async_comp():
            log.append("async-undo")

        saga = (
            Saga("test")
            .step("a", lambda: None, async_comp)
            .step("b", lambda: (_ for _ in ()).throw(RuntimeError("x")), lambda: None)
        )
        with pytest.raises(SagaExecutionError):
            await saga.execute()

        assert "async-undo" in log

    @pytest.mark.asyncio
    async def test_first_step_failure_no_compensation(self):
        from resilience.saga import Saga, SagaExecutionError
        log: list[str] = []

        saga = Saga("test").step(
            "a",
            lambda: (_ for _ in ()).throw(RuntimeError("immediate fail")),
            lambda: log.append("undo-a"),
        )
        with pytest.raises(SagaExecutionError):
            await saga.execute()

        assert log == []  # nothing completed, nothing to compensate

    @pytest.mark.asyncio
    async def test_returns_list_of_step_results(self):
        from resilience.saga import Saga
        saga = (
            Saga("test")
            .step("a", lambda: 1, lambda: None)
            .step("b", lambda: 2, lambda: None)
        )
        results = await saga.execute()
        assert results == [1, 2]

    @pytest.mark.asyncio
    async def test_compensation_failure_marks_not_compensated(self):
        from resilience.saga import Saga, SagaExecutionError
        log: list[str] = []

        def bad_comp():
            raise RuntimeError("compensation also fails")

        saga = (
            Saga("test")
            .step("a", lambda: log.append("a"), bad_comp)
            .step("b", lambda: (_ for _ in ()).throw(RuntimeError("step fail")), lambda: None)
        )
        with pytest.raises(SagaExecutionError) as exc_info:
            await saga.execute()

        assert exc_info.value.compensated is False


# ── Retry ─────────────────────────────────────────────────────────────────────

class TestRetryDecorator:

    def test_succeeds_on_first_try(self):
        from resilience.retry import with_exponential_backoff
        calls = {"n": 0}

        @with_exponential_backoff(max_attempts=3, min_wait=0.01, max_wait=0.05)
        def flaky():
            calls["n"] += 1
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 1

    def test_retries_on_transient_failure(self):
        from resilience.retry import with_exponential_backoff
        calls = {"n": 0}

        @with_exponential_backoff(max_attempts=3, min_wait=0.01, max_wait=0.05)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise IOError("transient")
            return "recovered"

        assert flaky() == "recovered"
        assert calls["n"] == 3

    def test_raises_after_max_attempts(self):
        from resilience.retry import with_exponential_backoff
        calls = {"n": 0}

        @with_exponential_backoff(max_attempts=2, min_wait=0.01, max_wait=0.05)
        def always_fails():
            calls["n"] += 1
            raise IOError("always bad")

        with pytest.raises(IOError):
            always_fails()
        assert calls["n"] == 2

    def test_does_not_retry_unmatched_exception(self):
        from resilience.retry import with_exponential_backoff
        calls = {"n": 0}

        @with_exponential_backoff(
            max_attempts=3, min_wait=0.01, max_wait=0.05,
            exceptions=(IOError,),
        )
        def wrong_exc():
            calls["n"] += 1
            raise ValueError("not retried")

        with pytest.raises(ValueError):
            wrong_exc()
        assert calls["n"] == 1
