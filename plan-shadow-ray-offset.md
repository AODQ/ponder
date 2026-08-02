# plan: scene-aware shadow/continuation ray offset

Status: revised. Step E (degenerate `maxDist` guard) has **landed**; everything
else is proposed.

Revision note: the first draft derived the offset from the scene bounding box.
That is dropped — see "the bounding box is not needed". The primary mechanism is
now a scale-aware `minT`, which is both smaller to implement and the better
precedented choice.

## Problem

Every ray that leaves a surface is nudged off it by a hardcoded `0.0005f`
world-space constant, and every trace uses a hardcoded `minT` of `0.00001f`.
No uniform, no scaling, no per-asset control. The same numbers serve a
unit-cube glTF model and bistro.

- too high — contact shadows detach ("peter-panning"), light leaks under
  object bases, small props stop occluding
- too low — shadow acne, surfaces self-shadowing into noise

### Current literal sites

All in `app/viewer/shaders/util-raytrace.glsl`:

| line | what |
|---|---|
| `:67` | `rayQueryInitializeEXT` `minT = 0.00001f` — **every** trace flows through here |
| `:449` | env NEE shadow origin, `±0.0005f` along geometric normal |
| `:549` | mesh NEE shadow origin, `±0.0005f` |
| `:555` | mesh NEE far end, `es.distance - 0.001f` — **guard landed**, value still fixed |
| `:777` | continuation ray after the bounce |
| `:1118` | subsurface walk, TIR continue-inside nudge |
| `:1154` | subsurface exit-vertex NEE shadow origin |
| `:1208` | subsurface walk, exit-side nudge |
| `:1249` | subsurface walk, exit-side nudge |
| `:643` | pre-walk pull-inside, `1e-6f` — separate, much smaller, leave alone |

`:777` is applied *after* the NEE blocks (comment at `:774-776`), so a shadow
origin is hit-point + 0.0005, not + 0.001. No double offset.

## What actually drives the required offset

The hit point is reconstructed at `:908` as `itOri = itOri + bsdfWo * rq.dist`.
Its error is roughly `ulp(|itOri|) + ulp(rq.dist)` ≈ `1.2e-7 × max(|coord|, dist)`
— linear in scale with a **1e-7 slope**.

That rules out deriving anything from scene extent:

- `boundsExtent * 5e-4` (reproduces today's 0.0005 at extent 1) has a 5e-4
  slope, ~4000× too steep. On a 200 m bistro that is a 10 cm offset.
- `boundsExtent * 2.5e-5` (calibrated to precision) gives 2.5e-5 at extent 1,
  20× smaller than today, reintroducing acne on models that work now.

The gap between those is the tell: **0.0005 was never a precision-driven
number**, so extrapolating it by scene scale extrapolates whatever it was
really compensating for.

## What other renderers do

Four families, and most production renderers use several at once:

1. **Conservative error bounds — pbrt.** Carries a per-hit error bound
   (`pError`, the `γₙ` analysis) through every intersection and transform, then
   offsets by `dot(abs(n), pError)`. Provably correct, no magic constants.
   Impossible here: fixed-function BVH traversal reports no error bounds.
2. **Integer/ULP origin stepping — Cycles, OptiX samples, RTG ch. 6.** Moves
   the origin point by stepping its float bit pattern. RTG uses 256 ulp, Cycles
   ~32 ulp — the two best-known implementations disagree 8× on the one constant
   that matters.
3. **Scaled `tmin` — Mitsuba.** Leaves the hit point alone, raises the ray's
   `mint` scaled by coordinate magnitude, with a larger epsilon for shadow rays.
4. **A user-facing bias knob — Arnold, RenderMan.** Scene- and per-object-level,
   artist tuned, because shipping assets violate the theory.

Nobody derives it from the scene bounding box. Everybody derives it from the
**hit point's own coordinate magnitude**, then hands artists an override.

## Design

### The bounding box is not needed

`utilTraceRayImpl` already receives `origin` (`:45`). It can derive the
magnitude itself:

```glsl
// self-intersection error tracks ulp of the hit point's coordinates, so
// scale the epsilon off the origin's magnitude rather than scene extent
f32 utilRayEpsilon(const f32v3 p) {
	return max(
		kR)ayEpsilonMin,
		kRayEpsilonRel * max(max(abs(p.x), abs(p.y)), abs(p.z))
	);
}
```

`kRayEpsilonRel = 3e-5f` (RTG's 256 ulp expressed as a relative factor),
`kRayEpsilonMin = 1e-5f` (near-origin fallback, RTG's `1/65536`).

No `SceneState::boundsExtent` plumbing, no uniform fill, no per-frame anything.
The earlier finding that the bbox is already maintained at the right cadence
(`scene-state.cpp:462-475`, inside `stage_render_rebuild_tlas`) still holds —
it is simply not needed for this.

### Tier 1 (primary): scale-aware `minT`

One change, inside `utilTraceRayImpl` at `:67`:

```glsl
/*minT*/ utilRayEpsilon(origin),
```

Zero call-site changes. This covers **all** traces at once, including the four
subsurface sites (`:1118/:1154/:1208/:1249`) that have no `hitDistance` in
scope and would otherwise need it threaded through `utilSubsurfaceWalk`.

Known weakness: `tmin` measures *along the ray*, so at grazing angles it buys
almost no perpendicular clearance — exactly where self-intersection is worst.
Hence tier 2.

Second weakness: it sees only `|origin|`, not the length of the ray that
produced it. A long ray terminating near the world origin is under-offset. If
that shows up, pass an optional length hint; do not pre-emptively plumb it.

### Tier 2: normal-direction origin offset, same epsilon

Keeps perpendicular clearance at grazing, which tier 1 cannot provide. Replace
the `0.0005f` literals with `utilRayEpsilon(itOri)` at `:449`, `:549`, `:777`,
`:1118`, `:1154`, `:1208`, `:1249`. Sign selection is unchanged — the existing
`crossedSurface` and `dot(itNormalGeometrical, es.direction) >= 0.0f` ternaries
stay exactly as they are; only the magnitude moves.

### Tier 3: per-instance override

`GpuResolveModelIndirect` (`resolve_pc.h:20-38`) is **per-TLAS-instance**, not
per-mesh — BLASes are shared across instances (`scene-state.hpp:81`) while the
models-indirect buffer has one entry per instance with its own `modelMatrix`.

This does not improve precision; tiers 1-2 already adapt per hit. It covers what
precision math cannot predict because it is authored, not numeric: coplanar
decal sheets, foliage/hair cards, self-intersecting or inverted meshes,
triangles thin enough that BLAS quantization dominates. Every production
renderer ships an equivalent.

A multiplier over the derived epsilon, defaulting to `1.0f`. Already in scope at
the shading vertex:

- `RayQueryResult.modelDrawIndex` — `:24`, populated at `:129`
- `hitModelDrawIndex` is a parameter of `utilIrradiance` (`:306`) and
  `utilIrradiancePropagate` (`:806`)
- with mesh NEE on, that vertex **already** fetches
  `models.data[hitModelDrawIndex]` (`:347` → `:202`), so it costs no extra load

Note it cannot reach tier 1: `utilTraceRayImpl` has no model index. Tier 3
multiplies the tier-2 origin offset only, which is the tier that matters for
authored-geometry cases anyway.

### Global uniform

Keep `f32 shadowRayOffsetScale` in `GpuGlobalExtended` (next to
`meshNeeEnabled`, `global_pc.h:86`) as a **debug multiplier over the whole
thing**, defaulting to 1.0 — not as a bbox-derived value. Its only job is
letting an ImGui slider sweep the epsilon during bring-up, which is the only
practical way to tell leak-from-too-high apart from acne-from-too-low by eye.

## Risk: this makes unit-scale models much less conservative

At `|p|` ≈ 1, `utilRayEpsilon` returns 3e-5 against today's 5e-4 — **17× smaller**.
A unit-scale glTF sits mostly in `[-1, 1]`, so most of the scene gets 1e-5 to
3e-5. Bistro at `|p|` ≈ 200 gets ~6e-3, i.e. larger than today.

That is the intended adaptive behaviour, but if 0.0005 has been quietly masking
something *other* than float error — BLAS quantization on thin triangles,
coplanar geometry, the BLAS-seam overlap seen in the Cornell box work — the
change will surface it as acne and read as "the new epsilon is broken" when it
is not.

**So land it behind the global multiplier and compare, not as a wholesale swap.**
Sweep the slider on a unit model first; if it needs > 1.0 to look clean, the
residual is not precision and tier 3 is the right home for it.

## Steps

1. **`utilRayEpsilon` + constants** — top of `util-raytrace.glsl` near
   `utilAlphaCoverage` (`:34`). Needs no new types.
2. **Tier 1** — `:67`, `/*minT*/ utilRayEpsilon(origin)`.
3. **Global multiplier** — `global_pc.h:86`; fold into `utilRayEpsilon`'s
   result at each use. `main.cpp` ~`:1160` fills it from an ImGui slider,
   default 1.0.
4. **Tier 2** — replace the seven `0.0005f` literals. Leave `:643` and the
   landed `:555` guard alone.
5. **Tier 3** — `f32 shadowRayOffsetScale` in `GpuResolveModelIndirect`,
   stamped where `emitterLightOffset/Count` already are
   (`scene-state.hpp:18-25`), default `1.0f`.
6. **Far end** — now that tier 2 is scale-aware, `:555`'s fixed `0.001f` should
   follow it: `es.distance - utilRayEpsilon(itOri) * 2.0f`. The `> 0.0f` guard
   already landed and still covers the degenerate case.
7. **Verify** — Cornell box tests (contact shadow at the box/floor seam), then
   bistro and a unit-scale model as the two ends of the range. Per the
   "verify new vars wired in" rule, grep both new names afterwards and confirm
   every site reads them, not just the declarations.

## Deferred

- **Stage authoring UI for tier 3** — a per-instance override that survives a
  reload needs a field in `stage::GltfInstance`, scene-file round-trip, and an
  outliner control. Build it when a specific asset forces it.
- **Per-mesh AABB derivation** — `mor.hpp:54` exposes only whole-scene
  `scene_bounds`. Tiers 1-2 make this redundant.

## Alternative to tier 2: RTG `offset_ray`

Wächter & Binder, Ray Tracing Gems ch. 6. Relocates the origin point by
stepping its float bit pattern a fixed number of ulp, so the offset scales with
magnitude by construction — same goal as `utilRayEpsilon`, done exactly rather
than with a relative constant.

GLSL port is direct (`floatBitsToInt`/`intBitsToFloat` are core), with one gap:
the shader type block (`util-mis.glsl:4-17`) defines `i32` and `i32v2` but not
`i32v3`.

It also cleans up the call sites, collapsing the sign ternaries into the call:

```glsl
const f32v3 shadowOrigin = utilOffsetRay(
	itOri,
	dot(itNormalGeometrical, es.direction) >= 0.0f
		? itNormalGeometrical : -itNormalGeometrical
);
```

And it makes the both-ends form available for free, since `EmitterSample`
already carries `position` and `normal` (`util-emitter.glsl:288-292`) — pbrt
offsets both endpoints for the same reason:

```glsl
const f32v3 lightPoint = utilOffsetRay(es.position, -es.normal);
const f32v3 toLight = lightPoint - shadowOrigin;
const f32 shadowMaxDist = length(toLight);
```

Reasons not to lead with it: it returns a point rather than a scalar, so it
cannot feed tier 1's `minT`, and it has a known discontinuity — the ulp path
only exceeds its own `1/65536` near-origin fallback once `|p| > 0.5`, so
crossing outward past `|p| = 1/32` the offset *drops* ~16×. Worth adopting for
t)ier 2 once tier 1 is in and measured.
