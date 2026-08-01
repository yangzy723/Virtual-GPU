# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added a real in-process `LocalScheduler` with memory admission, kernel/memcpy slot waiting, and no socket dependency.
- Added `VGPU_SCHEDULER_MODE=daemon|local`; retained `VGPU_LOCAL_MODE=1` as a compatibility alias.
- Added C++ local-scheduler tests, local-mode integration coverage, a reproducible pure-GEMM A/B benchmark harness.
- Extended memcpy scheduling coverage to generic, peer, 2D, and 3D Driver API copy paths, including async and `_v2` variants where applicable.
- Added event-based completion reporting for `cuLaunchKernel`, `cuLaunchKernelEx`, and async memcpy paths, with API-return fallback if private event tracking is unavailable.
- Added scheduling for CUDA 13 `cuMemcpyBatchAsync_v2`; older Batch signatures and 3D Batch remain pass-through.
- Delayed `cuMemFreeAsync` FREE accounting until the freeing stream reaches a private completion event, with API-return fallback.
- Added an explicit one-time stderr warning when the shim enters fail-open passthrough because the scheduler daemon is unavailable or unresponsive.
- Added scheduling for common `cuMemsetD*` and `cuMemsetD2D*` Driver API paths, including async variants.

### Changed
- Split shim scheduling into explicit `scheduler_backend`, `local_scheduler`, and `daemon_channel` components.
- Consolidated scheduler configuration and exact socket I/O into shared helpers used by local and daemon paths.
- Removed unused scheduler APIs/state and deduplicated proc-address routing and completion-symbol resolution.
- Documented local, daemon, and fail-open semantics plus benchmark provenance and limitations.
- Made benchmark help usable without importing Torch, added P95 output, and derived MPS hierarchy SM counts from the active device.
- Moved GreenCtx lifecycle/pooling/policy scenarios and the separately measured MPS/GreenCtx resident-memory evaluation to the companion CoGPU repository under `eval/overhead/`.

## [0.3.2] - 2026-05-03

### Changed
- Made memcpy scheduling always enabled for `cuMemcpyHtoD/DtoH/DtoD` (including Async paths); removed the optional control switch.

### Fixed
- Removed stale "optional memcpy" wording and aligned README/DESIGN with actual runtime behavior.

## [0.3.1] - 2026-05-03

### Fixed
- Reduced overclaim wording in project docs to reflect real behavior on degradation and symbol coverage.
- Aligned memcpy interception docs with implementation details, including dynamic `cuGetProcAddress` resolution paths.

### Changed
- Refactored shim wait loops to remove duplicated spin-wait logic.
- Added configurable shim wait budget via `GPU_SCHEDULER_WAIT_ITERS` (default: `200000`).

### Added
- Added synchronous memcpy symbols (`cuMemcpyHtoD/DtoH/DtoD` and `_v2`) to the intercepted-symbol set used by `cuGetProcAddress` routing.
- Added project version file (`VERSION`).

### Notes
- Kernel/memcpy completion accounting remains API-return based (not device-completion based).
