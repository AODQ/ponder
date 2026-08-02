# plan: offline shader compile daemon (`shaderd`)

Goals: faster shader edits, and **no renderer stall** for slang or for
Vulkan pipeline compiles. Hot reload works only while a `shaderd` is
running. The renderer only picks up finished artifacts.

Scope (user, 2026-11-20): compute only (the entry kernels of
`lib/ponder/shaders`). Graphics pipelines and other shader dirs keep the
in-process mtime reload. With no daemon running there is no hot reload; the
renderer keeps its current pipelines.

## status: implemented (2026-11-20)

```
./build/release/shaderd --root lib/ponder/shaders   # cache in <root>/.shader-cache
shaderd --status | --errors | --rebuild [entry|all] | --forget | --quit
```

- **Terminal UI (FTXUI)** when stdout is a terminal. It shows a table per
  pipeline: state with live slang/driver timers, slang and driver time,
  registers, spills, occupancy and ISA size, sortable with `s`. It also shows
  the log and the error pane of the selected pipeline. Keys: `r` / `R`
  rebuild one / all, `f` forget, `q` quit. Driver and validation-layer
  output (stderr) goes to `<cache>/shaderd-stderr.log` and appears in the
  log pane.
- **`--plain`, or stdout not a terminal:** the timestamped log lines only.
- **Statistics** come from `VK_KHR_pipeline_executable_properties`.
  Occupancy is estimated from registers and the warp limit (NVIDIA:
  64K-register file, `VK_NV_shader_sm_builtins` for warps per SM). The
  driver sets a flag bit above the low 32 bits of "Local Memory Size", so
  only the low half is used.
- **Always optimized** (2026-11-20): the runtime toggle is gone; vkof
  requests O1 and its in-process fallback compiles without `-O0`.

Code:
- `app/shaderd/main.cpp`: options, lock and heartbeat, watcher, manifest,
  batch builds, control requests, client mode
- `app/shaderd/monitor.*`: thread-safe view state (rows, live job stages,
  log, ui commands) shared by the poll loop, workers and ui
- `app/shaderd/tui.*`: the FTXUI front end
- `app/shaderd/compile.cpp`: slang API compile with a per-define-set module
  cache and a pool of global sessions
- `lib/vkof/{include/vkof,src}/shader-cache.*`: the contract shared by
  daemon and renderer (key, paths, artifact/manifest json, heartbeat)
- `lib/vkof/src/vkof.cpp`, section "shader cache (shaderd artifacts)":
  pipeline binary capture/load, startup resolve, per-frame swap, deferred
  pipeline destroy
- `lib/ponder/src/wavefront.cpp`: `vkof::shader_cache_configure` at the top
  of `wavefront_create`

## measured

| | before | with shaderd |
|---|---|---|
| slang, integrate, cold | ~1.5 s (slangc) | 1.1–1.9 s |
| slang, accumulate after an edit | ~0.7 s | 87 ms (9 modules reused) |
| driver compile, integrate rgb / spectral | 16 s / 48 s, render thread | same, in the daemon |
| renderer: load the 4 entry kernels | ~65 s | 25–250 ms from pipeline binaries |
| `wavefront nee scene: matches lambert's closed form` | 62 s | 8.2 s |

Renders from binaries are byte-identical to in-process compiles.

The driver compile is the real cost. Slang is small next to it.

## contract

Cache dir layout (all writes are temp file + rename):

```
.shader-cache/
  shaderd.lock              instance id + heartbeat (unix ms), every 1 s
  generation                u64, bumped after each artifact
  manifest/<key>.json       written by vkof: {entry, defines, optimize}
  artifacts/<key>.json      generation, ok, error, deps + content hashes
  artifacts/<key>.spv
  artifacts/<key>.pbin      VK_KHR_pipeline_binary data (absent: printf)
  requests/<id>.json        control: {cmd, arg, answer: null -> text}
```

- key = fnv-1a of (entry relative to root, sorted defines, optimize flag)
- freshness = every dependency's content hash matches the file on disk
- `.pbin` header holds `pipelineCacheUUID` + driver version + vendor/device
  id; on a mismatch, or if binary creation fails, vkof compiles the `.spv`
- `vkGetPipelineKeyKHR(device, nullptr)` is not used: the validation layer
  segfaults on the null create info

Renderer:
- creation: fresh artifact -> load (binary); stale + daemon alive -> wait
  for it; stale + no daemon -> compile in-process as before
- per frame (polls every 100 ms): newer artifact -> load on a background
  thread -> swap at frame end; old pipeline destroyed `kFramesInFlight + 1`
  frames later. failed artifact -> keep old pipeline, show the error
- `pipeline_destroy` now defers destruction the same way

## follow-ups

- **driver compile time**: 48 s for the spectral integrate kernel is the
  edit-to-swap latency now. Try optimized slang SPIR-V (`--rebuild` after
  toggling optimize) to see if the driver compiles it faster; otherwise
  look at kernel size.
- module blobs aren't persisted, so a daemon restart compiles slang cold
  (1–2 s per kernel; small next to the driver)
- shaders that use printf get no binary: the validation layer must
  instrument them, so the renderer compiles them from SPIR-V (in the
  background on reload, blocking at startup)
- stale manifest entries stay until `shaderd --forget`
- not built on Windows yet; the code uses only std (plus a `/proc/meminfo`
  read that falls back to "plenty" when missing)
- pre-existing, unrelated: headless devices enable `VK_KHR_swapchain`
  without `VK_KHR_surface` (validation error at startup)
