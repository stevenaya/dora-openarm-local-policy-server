# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to communicate with a local policy server."""

import argparse
import contextlib
from dataclasses import dataclass
import dora
import json
import os
import pyarrow as pa
import socket
import tempfile
import threading
import time


START_COMMANDS = {"start"}
STOP_COMMANDS = {"stop", "intervene", "quit"}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass
class _PreparedRequest:
    request: dict
    request_json: str
    event_ns: int
    ready_ns: int
    event_period_ms: float | None
    prepare_arrow_ms: float
    json_request_ms: float
    path: str
    reset: bool
    generation: int
    should_log_timing: bool


@dataclass
class _ResetLatch:
    last_generation: int = 0

    def pending(self, generation):
        return generation != self.last_generation

    def consume(self, generation):
        self.last_generation = generation


class _ArrowFilePool:
    def __init__(self, shared_dir, pool_size):
        self.shared_dir = shared_dir
        self.paths = []
        self.extra_paths = []
        self.next_index = 0
        for _ in range(max(0, int(pool_size))):
            fp = tempfile.NamedTemporaryFile(
                suffix=".arrow",
                dir=shared_dir,
                delete=False,
            )
            fp.close()
            self.paths.append(fp.name)

    @property
    def enabled(self):
        return bool(self.paths)

    def reserve_path(self, protected_paths=()):
        protected = {path for path in protected_paths if path}
        for _ in range(len(self.paths)):
            path = self.paths[self.next_index]
            self.next_index = (self.next_index + 1) % len(self.paths)
            if path not in protected:
                return path

        fp = tempfile.NamedTemporaryFile(
            suffix=".arrow",
            dir=self.shared_dir,
            delete=False,
        )
        fp.close()
        self.extra_paths.append(fp.name)
        return fp.name

    def prune_extra(self, keep, protected_paths=()):
        if not self.extra_paths:
            return
        protected = {path for path in protected_paths if path}
        removable = max(0, len(self.extra_paths) - int(keep))
        if removable <= 0:
            return
        kept = []
        for path in self.extra_paths:
            if removable > 0 and path not in protected:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path)
                removable -= 1
            else:
                kept.append(path)
        self.extra_paths = kept


def _format_ms(value):
    return "NA" if value is None else f"{value:.2f}ms"


def _new_prepare_state():
    return {
        "previous_observation_id": None,
        "last_event_ns": None,
        "timing_index": 0,
        "generation": 0,
    }


def _start_episode(state):
    state["generation"] += 1
    state["previous_observation_id"] = None
    return state["generation"]


def _update_observation_generation(observation, state):
    observation_id = max(observation.field("id").to_pylist())
    previous_id = state["previous_observation_id"]
    if state["generation"] == 0 or (
        previous_id is not None and observation_id < previous_id
    ):
        _start_episode(state)
    state["previous_observation_id"] = observation_id
    return state["generation"]


def _prepare_request(event, state, arrow_pool, generation, protected_paths=()):
    event_ns = time.perf_counter_ns()
    event_period_ms = (
        None
        if state["last_event_ns"] is None
        else (event_ns - state["last_event_ns"]) / 1e6
    )
    state["last_event_ns"] = event_ns
    state["timing_index"] += 1

    prepare_start_ns = time.perf_counter_ns()
    path = arrow_pool.reserve_path(protected_paths)
    record_batch = pa.RecordBatch.from_struct_array(event["value"])
    with pa.output_stream(path) as output:
        with pa.ipc.new_file(output, record_batch.schema) as writer:
            writer.write(record_batch)
    ready_ns = time.perf_counter_ns()

    return _PreparedRequest(
        request={
            "name": "inference",
            "data_path": path,
            "metadata": event["metadata"],
        },
        request_json="",
        event_ns=event_ns,
        ready_ns=ready_ns,
        event_period_ms=event_period_ms,
        prepare_arrow_ms=(ready_ns - prepare_start_ns) / 1e6,
        json_request_ms=0.0,
        path=path,
        reset=False,
        generation=generation,
        should_log_timing=(
            state["timing_every"] > 0
            and state["timing_index"] % state["timing_every"] == 0
        ),
    )


def _serialize_request(prepared, reset):
    start_ns = time.perf_counter_ns()
    prepared.reset = reset
    prepared.request["reset"] = reset
    prepared.request_json = json.dumps(prepared.request)
    prepared.json_request_ms = (time.perf_counter_ns() - start_ns) / 1e6


def _send_actions(node, actions, reset):
    if not actions["positions"]:
        return False
    metadata = {"interval": actions["interval"], "reset": reset}
    if "cutoff_hz" in actions:
        metadata["cutoff_hz"] = actions["cutoff_hz"]
    node.send_output(
        "actions",
        pa.array(actions["positions"], type=pa.list_(pa.float32())),
        metadata,
    )
    return True


def _print_local_timing(
    *,
    mode,
    prepared,
    t_send_start_ns,
    t_request_sent_ns,
    t_response_done_ns,
    t_response_parsed_ns,
    t_loop_done_ns,
    actions,
    request_period_ms=None,
    next_request_wait_ms=None,
    dropped_stale=False,
):
    timing = actions.get("timing") or {}
    print(
        "[local-policy timing] "
        f"mode={mode} "
        f"loop={(t_loop_done_ns - prepared.event_ns) / 1e6:.2f}ms "
        f"event_period={_format_ms(prepared.event_period_ms)} "
        f"request_period={_format_ms(request_period_ms)} "
        f"next_request_wait={_format_ms(next_request_wait_ms)} "
        f"event_to_send={(t_send_start_ns - prepared.event_ns) / 1e6:.2f}ms "
        f"queued={(t_send_start_ns - prepared.ready_ns) / 1e6:.2f}ms "
        f"prepare_arrow={prepared.prepare_arrow_ms:.2f}ms "
        f"json_request={prepared.json_request_ms:.2f}ms "
        f"write_flush={(t_request_sent_ns - t_send_start_ns) / 1e6:.2f}ms "
        f"response_wait={(t_response_done_ns - t_request_sent_ns) / 1e6:.2f}ms "
        f"socket_rtt={(t_response_done_ns - t_send_start_ns) / 1e6:.2f}ms "
        f"json_response={(t_response_parsed_ns - t_response_done_ns) / 1e6:.2f}ms "
        f"send_output={(t_loop_done_ns - t_response_parsed_ns) / 1e6:.2f}ms "
        f"positions={len(actions.get('positions') or [])} "
        f"reset={prepared.reset} "
        f"dropped_stale={int(dropped_stale)} "
        f"server_policy={_format_ms(timing.get('policy_ms'))} "
        f"server_ready={_format_ms(timing.get('response_ready_ms'))} "
        f"server_arrow={_format_ms(timing.get('arrow_read_ms'))} "
        f"server_parse={_format_ms(timing.get('parse_observations_ms'))} "
        f"server_mmap={timing.get('arrow_memory_map', 'NA')}",
        flush=True,
    )


def _main_dora(io, shared_dir):
    n_keep_data = 5  # TODO: Customizable?
    timing_every = max(0, _env_int("LOCAL_POLICY_TIMING_EVERY", 20))
    arrow_pool_size = max(0, _env_int("LOCAL_POLICY_ARROW_POOL_SIZE", 0))
    arrow_pool = _ArrowFilePool(shared_dir, arrow_pool_size)
    state = _new_prepare_state()
    state["timing_every"] = timing_every

    node = dora.Node()
    cond = threading.Condition()
    shared = {
        "latest": None,
        "reader_done": False,
        "error": None,
        "in_flight_path": None,
        "active": False,
        "generation": 0,
    }

    def reader_loop():
        try:
            for event in node:
                if event["type"] != "INPUT":
                    continue

                event_id = event["id"]
                if event_id == "command":
                    command = event["value"][0].as_py()
                    with cond:
                        if command in START_COMMANDS:
                            shared["generation"] = _start_episode(state)
                            shared["active"] = True
                            shared["latest"] = None
                        elif command in STOP_COMMANDS:
                            shared["active"] = False
                            shared["latest"] = None
                        else:
                            continue
                        cond.notify_all()
                    continue

                if event_id != "observation":
                    continue
                with cond:
                    if not shared["active"]:
                        continue

                generation = _update_observation_generation(event["value"], state)
                with cond:
                    shared["generation"] = generation
                    latest = shared["latest"]
                    protected = {
                        shared["in_flight_path"],
                        latest.path if latest is not None else None,
                    }

                prepared = _prepare_request(
                    event,
                    state,
                    arrow_pool,
                    generation,
                    protected,
                )
                with cond:
                    if shared["active"] and shared["generation"] == generation:
                        shared["latest"] = prepared
                        cond.notify()
        except BaseException as exc:
            with cond:
                shared["error"] = exc
                shared["reader_done"] = True
                cond.notify()
        else:
            with cond:
                shared["reader_done"] = True
                cond.notify()

    reader = threading.Thread(
        target=reader_loop,
        name="local-policy-observation-reader",
        daemon=True,
    )
    reader.start()

    request_reset = _ResetLatch()
    output_reset = _ResetLatch()
    last_request_sent_ns = None
    while True:
        wait_start_ns = time.perf_counter_ns()
        with cond:
            while (
                shared["latest"] is None
                and not shared["reader_done"]
                and shared["error"] is None
            ):
                cond.wait()
            if shared["error"] is not None:
                raise shared["error"]
            if shared["latest"] is None and shared["reader_done"]:
                break
            prepared = shared["latest"]
            shared["latest"] = None
            shared["in_flight_path"] = prepared.path
        wait_done_ns = time.perf_counter_ns()

        _serialize_request(
            prepared,
            request_reset.pending(prepared.generation),
        )
        send_start_ns = time.perf_counter_ns()
        request_period_ms = (
            None
            if last_request_sent_ns is None
            else (send_start_ns - last_request_sent_ns) / 1e6
        )
        io.write(prepared.request_json + "\n")
        io.flush()
        request_sent_ns = time.perf_counter_ns()
        last_request_sent_ns = request_sent_ns
        request_reset.consume(prepared.generation)

        response = io.readline()
        response_done_ns = time.perf_counter_ns()
        if not response:
            break
        actions = json.loads(response)
        response_parsed_ns = time.perf_counter_ns()

        with cond:
            dropped_stale = (
                not shared["active"]
                or shared["generation"] != prepared.generation
            )
        if not dropped_stale and _send_actions(
            node,
            actions,
            output_reset.pending(prepared.generation),
        ):
            output_reset.consume(prepared.generation)
        loop_done_ns = time.perf_counter_ns()

        with cond:
            if shared["in_flight_path"] == prepared.path:
                shared["in_flight_path"] = None
            latest = shared["latest"]
            protected = {
                shared["in_flight_path"],
                latest.path if latest is not None else None,
            }
        arrow_pool.prune_extra(n_keep_data, protected)

        if prepared.should_log_timing:
            _print_local_timing(
                mode="latest",
                prepared=prepared,
                t_send_start_ns=send_start_ns,
                t_request_sent_ns=request_sent_ns,
                t_response_done_ns=response_done_ns,
                t_response_parsed_ns=response_parsed_ns,
                t_loop_done_ns=loop_done_ns,
                actions=actions,
                request_period_ms=request_period_ms,
                next_request_wait_ms=(wait_done_ns - wait_start_ns) / 1e6,
                dropped_stale=dropped_stale,
            )


def main():
    """Communicate with a local policy server."""
    parser = argparse.ArgumentParser(
        description="Communicate with a local policy server"
    )
    parser.add_argument(
        "--socket",
        default=os.getenv("SOCKET"),
        help="The local socket to communicate",
        type=str,
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="dora-openarm-local-policy-server", dir="/dev/shm"
    ) as shared_dir:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(args.socket)
            with sock.makefile("rw") as io:
                _main_dora(io, shared_dir)


if __name__ == "__main__":
    main()
