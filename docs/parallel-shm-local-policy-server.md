# Parallel Shared-Memory Local Policy Bridge

## Scope

This document describes the `dora-openarm-local-policy-server` Dora node: how it
schedules observations, transports dense inputs through shared memory, talks to
a local policy process, handles episode resets, and forwards returned actions.

The node does not load or execute a model. Model-side CUDA/TensorRT scheduling,
feature lag, action decoding, and action-window selection belong to the policy
process. The compatible GR00T implementation is
`Openarm-GR00T/open_eval/parallel_policy_server.py`.

## Purpose

The node is intended for low-latency online control. Its main goals are:

- avoid serializing and reopening a large Arrow IPC payload for every
  multi-camera observation;
- continue receiving and preparing observations while one policy request is in
  flight;
- send the newest available observation instead of accumulating a stale FIFO
  queue;
- preserve policy response metadata, including action-chunk provenance, when
  publishing Dora actions.

The node does not compare consecutive states and does not skip inference when
observations are similar.

## Process Boundary

The Dora node and policy process communicate through a Unix domain socket. For
the SHM transport they must also share the same host and `/dev/shm` mount
namespace.

```text
Dora observer
     |
     v
reader thread -- copy dense fields --> latest pending SHM slot
                                            |
                                            v
socket loop -------- JSON request ------> local policy process
            <------- JSON actions -------
     |
     v
Dora actions
```

The node has two internal execution paths:

- the reader thread continues consuming Dora events and preparing observations;
- the socket loop sends one request and waits synchronously for its response.

Only one socket request is in flight. "Parallel" means input collection and
preparation overlap that request; it does not mean several policy RPCs execute
at once.

## Inputs and Output

The node consumes:

- `observation`: Arrow observation batches from the observer;
- `command`: `start`, `stop`, `intervene`, or `quit` lifecycle commands.

It publishes:

- `actions`: the returned joint-position chunk as an Arrow array, with response
  and bridge metadata attached.

The node does not interpolate or blend the returned action chunk. Those steps
belong to the action executor.

## Latest-Only Scheduling

The reader thread and socket loop share one pending `latest` request:

1. `start` activates a new generation and clears any pending request.
2. The reader prepares each active observation and stores it as `latest`.
3. A newer observation replaces the previous unsent `latest` request.
4. The socket loop takes `latest`, marks its transport resource in flight, sends
   the request, and waits for one response.
5. While it waits, the reader can replace the pending request repeatedly.
6. After the response, the socket loop immediately takes the newest pending
   observation, if one exists.

This bounds queue growth and favors observation freshness. An observation that
has not been sent may be replaced, but a request already running in the policy
process is not cancelled.

The actual request rate can therefore be lower than the observer rate. It is
also bounded by policy latency and any policy-side inference-rate limit.

## Shared-Memory Ring

Set `LOCAL_POLICY_TRANSPORT=shm_ring`; the node normalizes this to the
`shm_ring_v1` wire protocol. On the first observation it creates an
application-owned ring file in its temporary `/dev/shm` directory. The ring has
three page-aligned slots.

Each slot stores the dense observation fields:

- `position` as `float32[history, dof]`;
- each available supported camera as
  `uint8[history, height, width, 3]`.

The supported camera fields are:

- `camera_wrist_right`
- `camera_wrist_left`
- `camera_head_left`
- `camera_head_right`
- `camera_ceiling`

The task prompt and scalar metadata remain in the JSON request. Each dense-field
descriptor contains its offset, shape, dtype, and byte count.

### Slot ownership

The three slots cover the node's possible ownership states:

- one slot may belong to the in-flight request;
- one slot may hold the newest pending request;
- one slot remains available for the next observation copy.

The writer excludes the in-flight and pending slots when selecting a slot.
Replacing a pending request releases its old slot for reuse.

The policy response must return the request's sequence as `input_sequence`.
After validating this acknowledgement, the node knows the policy no longer
needs the in-flight input and can reuse that slot.

### Capacity

The first observation determines the fixed slot capacity. Smaller later payloads
are allowed. A later payload larger than the slot raises
`ShmRingCapacityError`; restart the node after increasing camera resolution,
history length, or the observation layout.

### Copy behavior

The writer exposes the Dora-backed Arrow list values as NumPy arrays where
possible, then performs one CPU copy into the selected ring slot. SHM removes
per-request Arrow-file serialization, opening, and policy-side file reads; it
does not make the entire input path zero-copy.

## Policy-Process Contract

A compatible policy process must:

1. accept newline-delimited JSON requests on the configured Unix socket;
2. map the supplied SHM ring path read-only, preferably once per connection;
3. validate each field descriptor before creating views over the selected slot;
4. finish using the input before returning its sequence as `input_sequence`;
5. return `positions`, `interval`, optional `cutoff_hz`, optional `timing`, and
   optional policy metadata.

The GR00T `policy_server.py` and `parallel_policy_server.py` implement this
contract. Their model execution can differ without changing this node. For
example, the parallel server may return an empty prefill response after reset;
the node's reset latches explicitly support that behavior.

## Wire Protocol

A shared-memory request has this shape:

```json
{
  "name": "inference",
  "transport": "shm_ring_v1",
  "shm": {
    "path": "/dev/shm/.../policy-observation-....ring",
    "ring_size": 67121152,
    "slot_count": 3,
    "slot_size": 22372384,
    "slot": 1,
    "sequence": 42,
    "payload_size": 21422080,
    "fields": {
      "position": {
        "offset": 0,
        "shape": [1, 16],
        "dtype": "<f4",
        "nbytes": 64
      }
    },
    "task_prompt": "Place the pillow into the pillowcase."
  },
  "metadata": {"timestamp": 1788512400000000000},
  "reset": false
}
```

The policy response contains `positions`, `interval`, optional `cutoff_hz`,
optional `timing`, optional `metadata`, and `input_sequence` for SHM
acknowledgement.

`LOCAL_POLICY_TRANSPORT=arrow_file` retains the compatible request path for a
policy process that cannot map the same shared-memory namespace.

## Reset and Generations

`stop`, `intervene`, and `quit` deactivate the node and clear the pending
observation. Responses from an inactive or older generation are discarded. An
observation ID moving backwards also starts a new generation, covering an
observer stream reset.

The node has separate request-side and output-side reset latches:

- the first request in a generation carries `reset: true` until the policy
  acknowledges it with `reset_applied` or a non-empty action response;
- the first non-empty Dora action output in that generation carries
  `reset: true` for downstream trajectory state.

This separation matters when the policy uses the first real observation only to
prefill a cache and returns no positions.

## Action Forwarding

The node publishes the response `positions` without changing the chunk. It
copies the policy response's `metadata`, then adds:

- `interval` from the response;
- the output-side `reset` flag;
- `cutoff_hz` when supplied by the policy.

Because policy metadata is passed through, fields such as `chunk_id`,
`generated_timestamp_ns`, episode identifiers, and future metadata additions do
not need dedicated forwarding code in this node.

## Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `SOCKET` / `--socket` | none | Unix socket exposed by the policy process. |
| `LOCAL_POLICY_TRANSPORT` | `arrow_file` | `arrow_file`, `shm_ring`, or `shm_ring_v1`. |
| `LOCAL_POLICY_TIMING_EVERY` | `20` | Print timing every N observations; `0` disables it. |
| `LOCAL_POLICY_ARROW_POOL_SIZE` | `0` | Reusable files in Arrow compatibility mode. |

Example Dora configuration:

```yaml
- id: policy-server
  build: pip install -e ../dora-openarm-local-policy-server
  path: dora-openarm-local-policy-server
  env:
    SOCKET: /dev/shm/policy-server.socket
    LOCAL_POLICY_TRANSPORT: shm_ring
    LOCAL_POLICY_TIMING_EVERY: "20"
  inputs:
    observation: observer/observation
    command: evaluation-ui/arm_command
  outputs:
    - actions
```

Start the listening policy process before this Dora node. If it runs in a
container, mount host `/dev/shm` at `/dev/shm` in the container so both the Unix
socket and ring path refer to the same namespace.

## Timing and Diagnosis

The `[local-policy timing]` line separates input freshness, transport cost, and
policy latency:

- `event_period`: interval between observations received by the reader;
- `prepare_input`: SHM copy or Arrow-file preparation time;
- `queued`: time a prepared observation waits behind the in-flight request;
- `event_to_send`: age of that observation when its request is sent;
- `write_flush`: socket request write time;
- `response_wait` / `socket_rtt`: time waiting for the policy response;
- `json_response`: response parsing time;
- `send_output`: Dora output time;
- `loop`: observation arrival through completion of Dora output;
- `request_period`: actual policy request cadence;
- `dropped_stale`: whether the response belongs to an inactive or older
  generation;
- `server_input`, `server_parse`, `server_rate_wait`, `server_policy`, and
  `server_ready`: timing values reported by the policy process.

A large `queued` value means observations arrive faster than the serialized
policy path can consume them. Latest-only replacement prevents this delay from
becoming an increasing FIFO backlog, but it does not shorten the request already
in flight.

## Limitations

- SHM is local-only; both processes must share a host and `/dev/shm` namespace.
- Ring capacity is fixed after the first observation.
- One slow policy request still blocks the next request from being sent.
- The node optimizes transport and freshness; it does not accelerate model
  computation itself.
- The SHM protocol assumes a trusted local policy process and does not provide
  authentication or cross-host coherency.
