# dora-openarm-local-policy-server

OpenArm's caller-side policy control: latest observations, request pacing,
execution boundaries, action windows and asynchronous chunk logging.

- Socket mode uses a separate model process and **SHM-only** observation transport.
- Direct Dora mode runs the same caller session with an in-process backend in
  that model's Python environment. Native Arrow is available on this path.
- Input borrowing is opt-in for supported backends on either transport.
- Neither mode blends or filters commands; those operations remain in the executor.

See [serving and transport](docs/serving.md) for the lifecycle, reset and wire contracts.
In the workspace, use `launch_inference.sh experiment.yaml` with the fixed
`openarm-policy-entry` dataflow node. Install the node together with the local
`packages/openarm-policy-runtime` source package; it is not assumed to be on PyPI.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
Copyright 2026 Enactic, Inc.

All participation follows our [Code of Conduct](CODE_OF_CONDUCT.md).
