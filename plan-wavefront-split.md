# Wavefront split — implementation plan

Written 2026-11-20. Goal: replace the integrate megakernel with small kernels
connected by queues, so that each kernel compiles quickly, an edit only
recompiles the kernels it touches, and divergent paths stop limiting each
other's occupancy.

Reference design: Laine, Karras, Aila, *Megakernels Considered Harmful:
Wavefront Path Tracing on GPUs*, HPG 2013. pbrt-v4's GPU integrator
(`src/pbrt/wavefront/`) is a complete open-source instance of the same idea.

---

## 1. Where we are

`entry/wavefront-integrate.slang` is a megakernel, despite its name. One thread
per pixel runs `samplePath`'s 8-bounce loop to completion. Inlined into that one
function:

| piece | where |
|---|---|
| camera ray, wavelength sample | entry `main` |
| transport stack: Schwarzschild → Bend (index field) → ParticipatingMedium (medium stack) → Minkowski | `path/transport-*.slang`, built in `main` for every path |
| geodesic stepping, RK4, Kerr metric, accretion disk emission | `transport-schwarzschild.slang` (757 lines) |
| surface vertex: material resolve, emission, emitter / ReGIR pdf for MIS | `implSamplePathHitEventSurface` |
| medium vertex: phase sample and evaluate | `samplePath`, `MediumScatter` case |
| NEE: light selection (power cdf or ReGIR) + visibility through the same transport stack | `path/nee.slang`, `light/regir.slang` |
| escape: environment radiance and pdf | `samplePath`, `Escape` case |
| path consumer (evaluate, or debug-record wrapping evaluate) | `path/path-consumer.slang` |

Threads find work through our own queues of path indices, consumed by
indirect dispatches. There is no ray tracing pipeline or shader binding
table: vkof only has `VK_KHR_ray_query`, and SER (driver-side reordering) does
nothing on Turing.

Driver compile times (shaderd, 2026-11-20, RTX 2080 Ti, driver 610.57):

| variant | slang | driver |
|---|---|---|
| integrate, rgb, O0 | 1.1 s | 16 s |
| integrate, spectral, O0 | 1.9 s | 48 s |
| integrate, spectral + debug path, O0 | 1.3 s | 216 s |
| accumulate (for scale) | 0.7 s | 0.02 s |

Slang is negligible next to the driver. My guess at why, not measured: the
driver's register allocation and scheduling grow faster than linearly with the
size of a single function, and everything above is one function. The debug
variant adds comparatively little code but takes 4.5× longer, which fits that.
§3 step 0 checks this before anything is restructured.

Runtime has the same root cause: the kernel's register count is set by its
heaviest branch (geodesic stepping), so flat-space paths run at the occupancy
of curved ones, and paths of different lengths keep warps partly idle.

---

## 2. Target

### Kernels

One bounce of every live path is a fixed sequence of dispatches. The host
loops it `bounceMax` times; each dispatch is indirect, sized by the queue it
consumes.

| kernel | consumes | produces |
|---|---|---|
| `wf-camera` | pixels | path state, ray queue |
| `wf-trace-flat` | ray queue (flat) | hit record, one event queue per event type |
| `wf-trace-curved` | ray queue (curved) | same |
| `wf-surface` | surface queue | next ray (ray queue), NEE request |
| `wf-medium` | medium-scatter queue | next ray, NEE request |
| `wf-escape` | escape + emission queues | radiance; path ends or continues (transparent emitter) |
| `wf-nee-select` | NEE requests | shadow-ray queue with light sample and pdfs (power cdf or ReGIR) |
| `wf-shadow-flat` / `wf-shadow-curved` | shadow-ray queue | MIS-weighted NEE radiance, added to the path |
| `wf-roulette` | live paths | survivors stay in the ray queue |
| `wf-accumulate` | (existing) | |

- **Flat and curved are separate kernels.** The geodesic code then only
  compiles into, and only costs registers in, the kernels that need it. The
  check that already exists in the transport ("defers to bend while the black
  hole is off or out of reach") becomes the router that picks the queue.
- **Per-material shading kernels later.** Once OpenPBR is the backend,
  `wf-surface` splits per backend and the surface queue is sorted by material.
- **Debug path.** Each kernel records its own stage when the path is the debug
  pixel, which replaces the separate debug megakernel variant.

### Data (transient buffers)

- **Path state, structure of arrays:** pixel, throughput β, accumulated Li,
  wavelengths, bounce and surface-bounce counters, previous-scatter record
  (`ImplPathScatterState`), and transport state (flat ori/dir/time/path length
  plus the medium stack). The sampler is stateless (pixel, sample index,
  dimension), so it stores nothing; this is the property that makes the split
  cheap.
- **Hit record per path:** the subset of `TraceTransportHit` the shading
  kernels read.
- **NEE / shadow record per path:** target, radiance, `pdfEmitFromNee`,
  `pdfBsdfFromNee`, ReGIR slot, debug.
- **Queues:** `u32` path indices plus an atomic counter each. A one-thread
  kernel turns the counters into `VkDispatchIndirectCommand`s.
- **Size:** estimate ~200–250 bytes per path. At 1920×1080 that is ~0.5 GB, so
  paths run in passes of a fixed pool (e.g. 1M paths), as pbrt-v4 does.

### Mapping from today's code

| today | after |
|---|---|
| `samplePath` loop | host loop over bounces; the loop body is split across kernels |
| `I_TraceTransport` (mutating struct per thread) | load from / store to path state at the start / end of each trace kernel |
| `implSamplePathHitEventSurface` | body of `wf-surface`, minus `implSampleNee` |
| `implSampleNee` | split: light selection in `wf-nee-select`, visibility in `wf-shadow-*` |
| `I_PathConsumer.vertex(vertex)` | see decision D2 |
| `PathConsumer_DebugRecord` | per-kernel recording (above) |

---

## 3. Steps

Each step leaves the image unchanged. Check: the wavefront unit tests
(`wavefront nee scene: *`) plus a byte compare of a fixed-spp headless render
before and after. The sampler is addressed by dimension, so the random numbers
do not change when the code moves.

Who does what: slang is yours. I do the vkof and C++ plumbing (indirect
dispatch, buffers, render graph wiring in `wavefront.cpp`) and measurements
through shaderd.

0. **Skipped (2026-11-20):** the split is wanted for runtime performance on
   complex scenes regardless of what it does for compile time.

   ~~**Measure.** Find which piece costs driver time before moving anything.~~
   Each item is a compile-only variant; shaderd prints the driver time.
   - O0 vs O1 SPIR-V: no win (§5).
   - Knockouts via defines in the entry or `samplePath`: Minkowski transport
     instead of Schwarzschild; NEE off; medium case off; ReGIR off.
   - `[noinline]` on `implSampleNee` and `traceTransport`. Unverified whether
     NVIDIA honours SPIR-V's DontInline; if it does, this is a near-free
     compile-time win and also confirms the size hypothesis.

   **Gate:** if one piece (likely the geodesic stepping) accounts for most of
   the time, step 3 alone may be enough.
1. **Done (2026-11-20). vkof:** `cmd_dispatch_indirect` (+ `_pushconst`),
   `cmd_fill_buffer`, `cmd_barrier_compute` (compute/fill → compute/indirect
   within one node). Test: `compute: indirect dispatch reads arguments
   written in the same node`.
2. **One bounce per dispatch.** The loop body becomes a kernel, with the host
   looping over bounces and path state in buffers. No queues yet; every pixel
   is a live path. This moves the hard part, serializing transport and scatter
   state, without changing control flow. It also shows how much of the
   compile time came from the driver unrolling 8 bounces.
3. **Trace split from shading,** with flat and curved trace kernels and the
   event queues.
4. **NEE split:** `wf-nee-select` and `wf-shadow-*` with the shadow queue.
   Do this after roadmap step 4 (connection interface): the shadow kernel is
   built on `visibilityQuery`'s return shape.
5. **Shading kernels per event** (surface / medium / escape).
6. **Path pool and regeneration,** if memory requires it (§2 size).
7. **Per-backend surface kernels with material sorting,** once OpenPBR lands.

Later, out of scope here: GPU-driven alternatives to host-looped indirect
dispatch, especially with sorting (persistent threads, work graphs, SER on
newer hardware).

### Balance

Requirement (user, 2026-11-20): no kernel is distinctly slower than the
others, either to compile or at runtime. A kernel that is splits further
along the cuts in §2.

Measured per kernel:
- **compile:** driver ms from shaderd (already logged)
- **runtime:** GPU ns per path processed. Raw ms is misleading, since trace
  handles every path and escape handles few.
- **registers and spills:** what limits occupancy, read from
  `VK_KHR_pipeline_executable_properties`

Tooling (vkof / shaderd, mine; build alongside step 2):
- GPU timestamps around each dispatch inside a node, summed per kernel
  across bounces. The profiler currently times whole nodes only.
- Queue counts read back per bounce, to turn kernel time into time per path.
- Register count and spill statistics per kernel in `shaderd --status`.
- Profiler shows per frame: dispatch count, which kernels ran and how often
  (user, 2026-11-20).

### Frame budget (user, 2026-11-20)

Predict each kernel's cost from previous frames and issue only as many
dispatches per frame as keep 60 fps; in the worst case a frame issues a
single dispatch. Paths then span frames ("partial wavefront runs"), which
needs:
- a per-path sample index (the sampler seeds from `global.sampleIndex`)
- accumulate adds only finished paths, and finished paths regenerate at the
  camera (step 6, pulled forward)
- a camera change flushes or finishes the in-flight paths

---

## 4. Decisions for you

- **D1 transport state.** `I_TraceTransport` is a mutating struct that lives
  in registers. Either each transport gets `load(pathIndex)` /
  `store(pathIndex)`, or the path state stores one flat union of all transport
  fields that every transport reads. The first keeps the layering; the second
  is simpler and is what pbrt-v4 does.
- **D2 consumer interface.** A vertex is complete only after its shadow ray
  returns. Either the consumer receives two calls per vertex (the BSDF-side
  terms at shading, the NEE term after shadow), or the vertex record is
  stored and the consumer runs in `wf-shadow-*`. MIS weights only need pdfs,
  which are known before visibility, so the two-call form works.
  `plan-path-infrastructure.md` §3 owns this interface, so the change belongs
  there.
- **D3 where the split sits in the roadmap.** Steps 0–2 are independent.
  Step 4 should follow roadmap step 4. ReSTIR (step 8) and path reuse (step
  10) want per-path records anyway, which this provides.

---

## 5. Results

**O0 vs O1** (2026-11-20, shaderd `--jobs 1`, so nothing else competes):

| variant | slang | driver | total | binary |
|---|---|---|---|---|
| rgb O0 | 1.0 s | 14.6 s | 15.6 s | 780 KiB |
| rgb O1 | 5.1 s | 17.0 s | 22.1 s | 2087 KiB |
| spectral O0 | 1.1 s | 48.7 s | 49.8 s | 965 KiB |
| spectral O1 | 9.6 s | 38.4 s | 48.0 s | 3674 KiB |

No win: O1's slang time cancels its driver saving on spectral, and rgb gets
slower. Stay on O0 for iteration. Spectral costs the driver 3× rgb for the
same control flow (4-wavelength loops and arrays), another sign that driver
time follows code size.
