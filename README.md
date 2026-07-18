# dora-openarm-local-policy-server

A [Dora](https://dora-rs.ai/) node that communicates with a local policy server for OpenArm.

## Input transports

`LOCAL_POLICY_TRANSPORT=arrow_file` keeps the compatible Arrow IPC file path.
`LOCAL_POLICY_TRANSPORT=shm_ring` copies dense policy inputs into a three-slot,
application-owned `/dev/shm` ring. The Unix socket then carries only the slot
descriptor, and a compatible policy server maps the ring once per connection.
This transport requires both processes to share the same host and `/dev/shm`
namespace; use a network-capable transport when moving inference off-host.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
