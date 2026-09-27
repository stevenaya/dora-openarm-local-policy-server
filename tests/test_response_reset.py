"""A model-requested restart must not be overwritten by the local reset latch."""

import pytest

from dora_openarm_local_policy_server.main import _send_actions


@pytest.mark.parametrize(
    "local_reset,model_reset,expected",
    [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, True),
    ],
)
def test_execution_reset_is_merged(local_reset, model_reset, expected):
    outputs = []

    class Node:
        def send_output(self, *args):
            outputs.append(args)

    response = {
        "positions": [[0.0] * 16],
        "interval": 33_333_333,
        "reset_applied": True,
        "metadata": {"reset": model_reset, "chunk_id": "one"},
    }
    assert _send_actions(Node(), response, local_reset)
    assert outputs[0][2]["reset"] is expected
    assert outputs[0][2]["chunk_id"] == "one"
    assert response["metadata"]["reset"] is model_reset
