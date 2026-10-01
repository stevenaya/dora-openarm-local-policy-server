# dora-openarm-local-policy-server

A [Dora](https://dora-rs.ai/) node that communicates with a local policy server for OpenArm.

## How it works

This node bridges Dora observations to a separate model process; it does not
load a model or blend actions. A reader thread prepares observations while the
socket loop waits for inference. Only one request is in flight; newer inputs
replace the unsent pending observation instead of building a FIFO backlog.
It does not skip inference based on state similarity.

Optional SHM transport avoids per-request Arrow-file serialization and reopening.
It copies dense inputs once into a three-slot, application-owned `/dev/shm`
ring; the socket carries descriptors and JSON responses. This improves transport
overhead and input freshness, not model computation, and is not end-to-end
zero-copy. Both processes must share the host and `/dev/shm` namespace.

## Usage

```yaml
- id: policy-server
  path: dora-openarm-local-policy-server
  env:
    SOCKET: /dev/shm/policy-server.socket
    LOCAL_POLICY_TRANSPORT: shm_ring
    POLICY_START_TIMEOUT_SEC: "600"
  inputs:
    observation: observer/observation
    command: evaluation-ui/arm_command
  outputs:
    - actions
    - status
```

| Setting | Default | Purpose |
| --- | --- | --- |
| `SOCKET` / `--socket` | Required | Local model's Unix socket. |
| `LOCAL_POLICY_TRANSPORT` | `arrow_file` | Arrow IPC files, or `shm_ring` / `shm_ring_v1`. |
| `POLICY_START_TIMEOUT_SEC` | `600` | Startup connection/readiness timeout in seconds. |
| `LOCAL_POLICY_TIMING_EVERY` | `20` | Timing log frequency; `0` disables logging. |
| `LOCAL_POLICY_ARROW_POOL_SIZE` | `0` | Preallocated reusable files in Arrow mode. |

## Readiness and lifecycle

The node joins Dora and publishes `status: loading` before connecting. On the
same connection used for inference it sends `{"ping": true}` and requires
`{"ready": true}` before publishing `ready`. The model must finish its warmup
before acknowledging. **Both transports require this handshake**; a socket file
alone is insufficient. Status metadata contains `message` and a Unix-nanosecond
`timestamp`. Errors report `error`; there is no automatic reconnect or replay.

Connect `status` to the evaluation UI's `policy_status` input and enable its
`WAIT_FOR_POLICY_READY=true` setting. Start commands received while loading are
ignored, not deferred. After readiness, `start` activates a new generation;
`stop`, `intervene`, and `quit` deactivate it and clear pending inputs.
Observation IDs moving backwards also advance the generation. A generation
check rejects stale responses; an in-flight inference is not cancelled by Stop.
Idle sockets are not polled, so idle disconnects surface on the next request.

## Model protocol

Requests and responses are newline-delimited JSON on the Unix socket. Requests
contain `name: inference`, `metadata`, `reset`, and either `data_path` (Arrow)
or `transport: shm_ring_v1` plus `shm` (ring descriptor). Metadata must be JSON
serializable, including timestamps; Python `datetime` values need conversion.

SHM carries `position` and supported camera arrays with history rows, plus the
newest `task_prompt`. Camera metadata supplies `<camera>.height/width`; arrays
are expected as float32 positions and flattened RGB uint8 images. The descriptor
contains path, slot, sequence, and per-field offset/shape/dtype/byte count.
The model must finish using the input before echoing its sequence as
`input_sequence`. Pending and in-flight slots are protected from overwrite.
Capacity is fixed by the first observation; restart after increasing payload size.

Responses contain `positions` and `interval` (nanoseconds), optionally
`metadata`, `cutoff_hz`, `timing`, and `reset_applied`. Returned chunks and their
metadata, including `chunk_id`, are forwarded without interpolation or slicing.
Empty prefill responses produce no action output. Request reset stays pending
until acknowledged by `reset_applied` or nonempty actions; output reset stays
pending until the first nonempty chunk. A model's `metadata.reset` is ORed with
the bridge's output reset rather than overwritten.

Timing logs distinguish input preparation, queued time, socket round trip,
output publication, and optional model-reported timings. A slow in-flight
request still delays the next request. SHM is a trusted local protocol, not a
network transport or a generic serialization of every observation field.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
