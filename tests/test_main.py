"""Tests for local policy request generation and reset delivery."""

import json

import pyarrow as pa

from dora_openarm_local_policy_server.main import (
    _PreparedRequest,
    _ResetLatch,
    _new_prepare_state,
    _send_actions,
    _serialize_request,
    _start_episode,
    _update_observation_generation,
)


def _observation(*ids):
    return pa.StructArray.from_arrays(
        [pa.array(ids, type=pa.int64())],
        names=["id"],
    )


def _prepared_request(generation):
    return _PreparedRequest(
        request={"name": "inference"},
        request_json="",
        event_ns=0,
        ready_ns=0,
        event_period_ms=None,
        prepare_arrow_ms=0.0,
        json_request_ms=0.0,
        path="observation.arrow",
        reset=False,
        generation=generation,
        should_log_timing=False,
    )


class _FakeNode:
    def __init__(self):
        self.outputs = []

    def send_output(self, *args):
        self.outputs.append(args)


def test_start_advances_generation_once_before_first_observation():
    """A command start and its first observation belong to one generation."""
    state = _new_prepare_state()

    assert _start_episode(state) == 1
    assert _update_observation_generation(_observation(0, 0), state) == 1

    assert _start_episode(state) == 2
    assert _update_observation_generation(_observation(0, 0), state) == 2


def test_latest_observation_id_decrease_is_reset_fallback():
    """A decreasing latest observation ID starts a fallback generation."""
    state = _new_prepare_state()
    _start_episode(state)

    assert _update_observation_generation(_observation(0, 10), state) == 1
    assert _update_observation_generation(_observation(0, 11), state) == 1
    assert _update_observation_generation(_observation(0, 0), state) == 2


def test_request_reset_stays_pending_when_latest_observation_is_replaced():
    """Replacing an unsent latest request must not consume its reset."""
    latch = _ResetLatch()
    latch.consume(1)

    assert latch.pending(2)
    assert latch.pending(2)

    prepared = _prepared_request(2)
    _serialize_request(prepared, latch.pending(prepared.generation))
    latch.consume(prepared.generation)

    assert json.loads(prepared.request_json)["reset"] is True
    assert not latch.pending(2)


def test_output_reset_is_consumed_only_after_nonempty_actions():
    """An empty response must not consume the executor-facing reset."""
    node = _FakeNode()
    latch = _ResetLatch()

    empty = {"interval": 1, "positions": []}
    if _send_actions(node, empty, latch.pending(1)):
        latch.consume(1)
    assert latch.pending(1)
    assert node.outputs == []

    actions = {"interval": 1, "positions": [[1.0, 2.0]]}
    if _send_actions(node, actions, latch.pending(1)):
        latch.consume(1)
    assert not latch.pending(1)
    assert node.outputs[0][2]["reset"] is True
