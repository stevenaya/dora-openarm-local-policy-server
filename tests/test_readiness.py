"""Readiness uses the inference socket and never admits a premature Start."""

import json
import queue
import socket
import threading

import pyarrow as pa
import pytest

from dora_openarm_local_policy_server import main as bridge


def command(value):
    return {
        "type": "INPUT",
        "id": "command",
        "value": pa.array([value]),
        "metadata": {},
    }


class Node:
    def __init__(self, mode):
        self.mode, self.events, self.outputs = mode, queue.Queue(), []
        self.early_consumed, self.ready = threading.Event(), threading.Event()
        self.failed = threading.Event()

    def next(self, timeout):
        try:
            event = self.events.get(timeout=timeout)
            if event.get("id") == "command":
                self.early_consumed.set()
            return event
        except queue.Empty:
            return {
                "type": "ERROR",
                "error": "Timeout event stream error: Receiver timed out",
            }

    def send_output(self, name, values, metadata):
        self.outputs.append((name, values, metadata))
        if name == "status" and values[0].as_py() == "loading":
            self.events.put(
                {"type": "STOP"} if self.mode == "stopped" else command("start")
            )
        elif name == "status" and values[0].as_py() == "ready":
            self.ready.set()
            if self.mode in {"ok", "cancel-in-flight"}:
                self.events.put(
                    {
                        "type": "INPUT",
                        "id": "observation",
                        "value": observation(),
                        "metadata": {"timestamp": 1},
                    }
                )
                # With no accepted Start, the preceding observation must be ignored.
                self.events.put(command("start"))
                self.events.put(
                    {
                        "type": "INPUT",
                        "id": "observation",
                        "value": observation(),
                        "metadata": {"timestamp": 2},
                    }
                )
        elif name == "status" and values[0].as_py() == "error":
            self.failed.set()
        elif name == "actions":
            self.events.put({"type": "STOP"})


def observation():
    return pa.StructArray.from_arrays(
        [pa.array([1]), pa.array([[0.0] * 16], type=pa.list_(pa.float32()))],
        names=["id", "position"],
    )


@pytest.mark.parametrize("mode", ["ok", "rejected", "disconnected", "cancel-in-flight"])
def test_socket_readiness(monkeypatch, tmp_path, mode):
    node, requests, failures = Node(mode), [], []
    monkeypatch.setattr(bridge.dora, "Node", lambda: node)
    monkeypatch.setenv("LOCAL_POLICY_TIMING_EVERY", "0")
    path = str(tmp_path / "policy.sock")

    def server():
        try:
            assert node.early_consumed.wait(2)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(path)
                listener.listen(1)
                conn, _ = listener.accept()
                with conn, conn.makefile("rw") as io:
                    requests.append(json.loads(io.readline()))
                    io.write(
                        json.dumps({"ready": mode != "rejected", "positions": []})
                        + "\n"
                    )
                    io.flush()
                    if mode == "ok":
                        requests.append(json.loads(io.readline()))
                        io.write(
                            json.dumps(
                                {"positions": [[0.0] * 16], "interval": 33333333}
                            )
                            + "\n"
                        )
                        io.flush()
                        assert io.readline() == ""
                    elif mode == "cancel-in-flight":
                        requests.append(json.loads(io.readline()))
                        node.events.put({"type": "STOP"})
                        assert io.readline() == ""
                    elif mode == "disconnected":
                        assert node.ready.wait(2)
                if mode == "disconnected":
                    assert not node.failed.wait(0.35)
                    node.events.put(command("start"))
                    node.events.put(
                        {
                            "type": "INPUT",
                            "id": "observation",
                            "value": observation(),
                            "metadata": {"timestamp": 3},
                        }
                    )
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=server, daemon=True)
    worker.start()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        if mode in {"ok", "cancel-in-flight"}:
            bridge._main_dora(sock, tmp_path, socket_path=path)
        else:
            with pytest.raises((ConnectionError, RuntimeError)):
                bridge._main_dora(sock, tmp_path, socket_path=path)
    worker.join(2)
    assert not worker.is_alive() and not failures
    assert requests[0] == {"ping": True}
    statuses = [value[0].as_py() for name, value, _ in node.outputs if name == "status"]
    if mode == "ok":
        assert statuses == ["loading", "ready"]
        assert len(requests) == 2 and requests[1]["metadata"]["timestamp"] == 2
        assert requests[1]["reset"]
        assert node.outputs[-1][0] == "actions" and node.outputs[-1][2]["reset"]
    elif mode == "cancel-in-flight":
        assert statuses == ["loading", "ready"]
        assert len(requests) == 2 and requests[1]["reset"]
        assert not any(name == "actions" for name, _, _ in node.outputs)
    else:
        assert statuses == (
            ["loading", "error"]
            if mode == "rejected"
            else ["loading", "ready", "error"]
        )
        assert not any(name == "actions" for name, _, _ in node.outputs)


@pytest.mark.parametrize("mode", ["timeout", "stopped"])
def test_startup_timeout_or_cancel(monkeypatch, tmp_path, mode):
    node = Node(mode)
    monkeypatch.setattr(bridge.dora, "Node", lambda: node)
    monkeypatch.setenv("POLICY_START_TIMEOUT_SEC", "0" if mode == "timeout" else "1")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        if mode == "timeout":
            with pytest.raises(TimeoutError):
                bridge._main_dora(
                    sock, tmp_path, socket_path=str(tmp_path / "absent.sock")
                )
        else:
            bridge._main_dora(sock, tmp_path, socket_path=str(tmp_path / "absent.sock"))
    expected = ["loading", "error"] if mode == "timeout" else ["loading"]
    assert [v[0].as_py() for n, v, _ in node.outputs if n == "status"] == expected
