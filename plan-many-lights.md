# Many-lights NEE via ReGIR — implementation plan

Successor to `plan-nee.md`, which built the estimator — one combined light list, exact
forward/backward pdf agreement — and deliberately left *selection* at a flat power CDF
(its §3.6). This plan replaces the selection strategy with **ReGIR**: a world-space grid
of light reservoirs, resampled at shading time.

> **Revised 2026-11-16 for the wavefront core (`lib/ponder/shaders/`).** The plan was
> first written against the old pipeline (`lib/ponder/shaders-old/`, `app/viewer/shaders/`).
> §R below is the current state and **overrides** the body wherever they disagree. Body
> sections that were rewritten are marked *(rev)*. Old identifiers that remain in
> unrevised prose map to new ones through the table in §2.1.

Written to be implemented from. Formulas are stated in the form you should code, ponder's
own identifiers are used throughout, and §6 is the part that has no precedent in either
published source — it is where ponder's progressive accumulator breaks assumptions the
papers make silently.

Division of labour: **you write `lib/` and `app/`; I write `unit-test/`.** §9 is mine and
is specified as a contract, not a suggestion — several of those tests are the only
instrument that can see the failure modes in §6 and §7.

---

## R. State against the wavefront core (2026-11-16)

### R.1 What already exists — in the old pipeline only

P1 and most of P2 were built on the old pipeline, and only there:

| piece | where | contents |
|---|---|---|
| layouts + constants | `shaders-old/util-regir-shared.h` | `GpuRegirCellKey`, `GpuRegirCell`, `GpuRegirReservoir`; `skRegirNormalResolution 4` (16 octahedral buckets + `skRegirNormalBucketNone`), `skRegirMaxProbesPerCell 32`, `skRegirReservoirCount 64`, `skRegirCandidateCount 32` |
| grid + reservoir | `shaders-old/util-regir.slang` | `utilRegirGridCoord`, `utilRegirNormalBucket`, `utilRegirCellHashKey`, `utilRegirInsert` (CAS on a separate guard buffer), `utilRegirLookup`, `utilRegirReservoir{Init,Update,Finalize}`, `utilRegirSelect` (uniform pick of 1 of 64 reservoirs, no shading-time resample, returns `slot`) |
| fill | `shaders-old/util-regir-fill.slang`, `app/viewer/shaders/regir-fill.slang` | `utilRegirTargetNonCanonical`, `utilRegirFillReservoir`; one workgroup per slot, one thread per reservoir, groupshared reduction to `normalizationNonCanonical` |
| clear | `app/viewer/shaders/regir-clear.slang` | |
| buffers | `app/viewer/scene-state.cpp:295` `regir_grid_ensure`; `sRegirEnabled = false` at `main.cpp:963` | guard / cell / reservoir buffers per scene-state |
| tests | `unit-test/src/test-regir.cpp` (tier 9, 7 cases), `test-regir-fill.cpp` (tier 10, 7 cases) | compiled with `PONDER_SHADER_DIR = shaders-old/` |

The wavefront core has none of it. **P1 and P2 are a port, not new work.** The math and
the layouts carry over unchanged. The module plumbing, buffer ownership and render-graph
wiring are new.

### R.2 The regression suite does not cover the new core

Every NEE/MIS test (`test-emitter-nee-scenes.cpp`, `test-env-mis.cpp`,
`test-emitter-sampling.cpp`, the tier 9/10 ReGIR tests) compiles against
`shaders-old/` through `util-raytrace`. **No unit test drives `vkof::samplePath`.**
§2.4's premise ("the tier 0-3 suite stays green untouched") therefore protects the old
pipeline, not the code ReGIR will be wired into. Building a new-core harness is phase P-1.

### R.3 New-core MIS defects to fix before ReGIR touches it

Read off `path/sample-path.slang` and `path/nee.slang`. Each sits on a line §4.4 rewrites,
so fix them first and separately. Otherwise a ReGIR bias cannot be told apart from an
existing one.

1. **Backward pdf written to the wrong field.** `sample-path.slang:230` assigns the
   emitter's solid-angle density to `vertex.pdfBsdfFromHit`, overwriting the bsdf pdf set
   at :53. `vertex.pdfEmitFromHit` stays `0`, so `implPathVertexWeightHit` returns `1`.
   BSDF-sampled hits on mesh emitters then get full weight, while NEE is still weighted by
   `balanceHeuristic(pdfEmitFromNee, pdfBsdfFromNee) < 1`. **Mesh emitters are
   over-counted whenever mesh NEE is on.** The fix assigns `vertex.pdfEmitFromHit`. The
   escape branch at :144 already does this correctly.
2. **Emitter-side cosine uses different normals.** The forward side
   (`nee.slang` → `emitterSample.normal = geometry.geometricNormal`) uses the geometric
   normal. The backward side (`sample-path.slang:226`) uses `geometry.shadingNormal`.
   These disagree on any emitter with smooth normals. Use `geometricNormal` on both sides.
3. **Medium NEE gate.** The `MediumScatter` case gates the whole NEE contribution on
   `envNeeEnabled` (`sample-path.slang:~97`), while `implSampleNee` gates mesh samples on
   `meshNeeEnabled`. With env NEE off and mesh NEE on, a mesh light sampled at a medium
   vertex is dropped. Its backward weight still uses `previousScatter.neeAvailable = false`,
   so the result is consistent (no bias), but mesh NEE in media is silently off. Gate on
   `meshNeeEnabled || envNeeEnabled`, as the surface case effectively does.

### R.4 Things that changed in ReGIR's favour

- **The sampler already returns area density.** `emitterSample` (`light/sample.slang:28`)
  picks a meshlet entry from the power CDF, a triangle uniformly inside it, and a point
  uniformly on the triangle. `EmitterSample.pdf` is per area for meshes. There is no
  hybrid spherical branch, so **§3.4 is moot**.
- **Full-frame dispatch.** `wavefront_integrate` dispatches `16×16` over
  `width × height`, and passes are render nodes. The tile-vs-fill conflict in **§6.2 is
  gone**: the fill is one more node before integrate.
- **Emitter granularity is unchanged.** The CPU builder is still
  `emitter-tables.cpp` (one `GpuEmitterLight` per emissive meshlet), read in-shader as
  `vkof::SceneEmitterLight` (`types/scene-light.slang`). §0's counts hold. The env entry
  is appended **last** (`emitter_lights_append_environment`, `scene-state.cpp:248`), and
  §2.3 now depends on that.
- **No delta lobes or subsurface in the new core yet.** It has only
  `MaterialBackend_Diffuse` with cosine-hemisphere sampling, and `isDeltaDirac` is always
  `false`. §6.4 and §6.5 have no site until track A lands.

---

## 0. Target and verdict

`bistro.gltf`, counted from the file:

| quantity | value |
|---|---|
| mesh-bearing nodes (instances) | 2 909 |
| instanced triangles | 4 209 006 |
| **emissive instances** | **115** |
| **emissive instanced triangles** | **76 248** |
| emissive meshlets, ~64 tri/meshlet | **~1 200** |

All ~1 200 emitter entries sit under **one `modelDrawIndex`** (the viewer loads one
`mor::Scene` per file). Structurally these are string lights and bulbs: small, scattered
along a street, and *mostly occluded from any given shading point*.

The flat CDF picks by `area x luma(emission)` and nothing else, so from a point under one
awning ~99% of picks land across the street — wrong hemisphere, behind a wall, or 30 m
away. Every one is a shadow ray returning zero. The estimator stays unbiased; it just
converges at roughly the rate of no light sampling at all. No existing test sees this,
because every tier 0-3 test gates on the **mean**.

**Why ReGIR and not a light BVH.** In the one published head-to-head on this scene
(Clabault 2025), a light tree with adaptive tree splitting measured *slightly worse than
power sampling* — 13.4 s vs 12.4 s — while ReGIR reached 5.79 s. Bistro's difficulty is
**visibility**, which no purely geometric importance function models. A light BVH bounds
`power / d^2` and cone overlap; it cannot know there is a wall in the way. ReGIR's target
function can, because it is evaluated per-cell where a visibility term is affordable.

Caveat on that evidence, stated once: it is a single self-reported benchmark from an
independent open-source renderer ([HIPRT-Path-Tracer](https://github.com/TomClabault/HIPRT-Path-Tracer)),
not production consensus — production *offline* renderers all use light trees. It is
weighted heavily here because that renderer is architecturally very close to ponder
(interactive GPU PT, progressive accumulation, unbiased, no denoiser). P0 reproduces the
comparison locally rather than inheriting either number.

**Consequence of going ReGIR-first: no light BVH is needed at all.** The grid fill needs
*some* candidate distribution and the existing power CDF is a perfectly good one — that is
the RTG2 baseline configuration. So this plan requires **zero changes to `mor`**, no
meshlet AABBs, no normal cones, no BVH build, no refit, no bitTrail. The candidate
generator is `emitterSelection` (`light/emitter.slang:20`) as it stands today, with its
`ξ` remapped onto the mesh range (§2.3). A light BVH remains available
later as a candidate-quality improvement (§10.5) and nothing here forecloses it.

---

## 1. ReGIR in ponder's terms

Two stages of resampled importance sampling (RIS), separated by a world-space cache.

**Stage 1 — fill.** Space is bucketed into cells (hash grid, keyed on quantized position
*and* quantized normal). Each cell has a representative point `P_c` and normal `N_c`. For
each of `R` reservoirs in the cell, draw `L` candidate light samples from a cheap source
distribution `p` (the power CDF), and stream them through weighted reservoir sampling
against a target function `hat_h` evaluated **at `P_c`**. Store the survivor plus its
unbiased contribution weight `W`.

**Stage 2 — shade.** A shading point hashes into its cell, pulls `S` reservoirs, resamples
them against its own actual BSDF and normal, and traces one shadow ray to the survivor.

The cost structure is the entire point: "evaluate ~1 200 lights properly" happens once per
cell and is amortized over every path vertex that lands there, instead of being redone per
vertex.

### 1.1 Measure convention — decide this once, on line one

**ReGIR works in area measure. ponder's NEE works in solid angle.** `hat_h` contains
`cos(theta_L) / d^2` explicitly, and the stored sample is a *point on a light*, so area is
the natural domain and fighting it produces Jacobians in six places.

Adopt: **everything inside ReGIR is area measure; convert exactly once, at the point of
contribution and at the point of MIS.**

```
contribution = f_r * L_e * V * G * W_Y                  // area measure
G            = cos(theta_s) * cos(theta_L) / d^2

p_light_omega(omega) = p_light_area(x) * d^2 / cos(theta_L)   // for MIS only
```

*(rev)* `emitterSample` already returns area density for mesh entries (§R.4). The one
conversion to solid angle stays where it is today, in `implSampleNee`:
`pdf_omega = pdf_area / (visibility.areaJacobian * cos(theta_L))`. `areaJacobian` is
`1/d^2` on the flat transport and the curved-path equivalent on the others. So ReGIR gets
lensing and media for free, as long as nothing writes a literal `d^2`.

### 1.2 The formulas

Candidates `X_1..X_M` from source pdf `p`. Resampling MIS weights `m_i` (see below).

```
w_i  = m_i(X_i) * hat_h(X_i) * W_{X_i}        // W = 1/p(X_i) for a fresh candidate
Y    = X_i chosen with probability w_i / sum(w)
W_Y  = (1 / hat_h(Y)) * sum_i(w_i)
```

`W_Y` is an unbiased estimate of `1 / p_true(Y)`. **At shading you multiply by `W_Y`; you
never divide by a pdf.** With uniform `m_i = 1/M` this collapses to the classic RIS form
`W_Y = (1/M) * sum(hat_h(X_i)/p(X_i)) / hat_h(Y)`.

**The normalization factor is free.** `Norm = integral hat_h(x) dA(x)` over all light
surfaces is estimated by `(1/M) * sum_i hat_h(X_i)/p(X_i)` — *exactly the sum the fill loop
already computes*. Accumulate it per cell during fill and store it alongside the
reservoirs. No extra pass. Then:

```
p_regir_area(x) = hat_h(x) / Norm
```

which is the density you need for MIS weights (§5.3).

### 1.3 Target functions

Two, per Clabault, and the second is not optional (§7 trap 21):

```
hat_h_N(x) = rho(P_c, w_o, w_i) * L_e(x) * V_approx(P_c <-> x) * cos(theta) * cos(theta_L) / |P_c - x|^2
hat_h_C(x) = L_e(x) / |P_c - x|^2
```

`hat_h_N` (non-canonical) is the good one — it can be zero. `hat_h_C` (canonical) is
deliberately crude and **can never be zero for a light that can contribute**, which is what
guarantees every emitter stays reachable.

Build `hat_h_N` up incrementally and measure each term (Clabault's ablation order):
power → `/d^2` → `cos(theta)` → `cos(theta_L)` → visibility. Ship without the visibility
term first; it is the largest single win but also the most machinery (§8, P5).

`rho` at fill time has no real `w_o` — the cell has no view direction. Use the cosine
term alone initially and treat a BSDF factor as an optional refinement; Clabault reports
limited benefit and it is wrong at secondary hits by construction.

*(rev)* **`L_e` in `hat_h` is a scalar and independent of wavelength.** The fill pass has
no `λ`, and the backward pdf must reproduce `hat_h` exactly. Do what the old
`utilRegirTargetNonCanonical` did: take it from the entry's precomputed
`SceneEmitterLight.radiance` (`emissionColor * emissionLuminance`), not from the textured
material. It costs no texture fetch in the fill. The backward side has `lightIndex` from
`emitterIndexFromHit` anyway, so both sides read the same `f32v3`, and there is one less
way for them to disagree. The *contribution* still evaluates the real material at the
sampled point (`implSampleNee` does so today), so a textured emitter costs variance, not
correctness.

```slang
// light/regir.slang -- the only place hat_h reads emission; fill and backward both call it
func implRegirEmissionLuma(light: vkof::SceneEmitterLight) -> f32 {
	// rec709, the same weights as luma() in emitter-tables.cpp:9
	return dot(light.radiance, f32v3(0.2126f, 0.7152f, 0.0722f));
}
```

Textured `L_e` in `hat_h` is a later refinement. If it is ever done, it has to change in
the fill and in `regirPdfArea` in the same commit.

`hat_h_N` uses the straight-line `cos(theta_L) / |P_c - x|^2` even under a bent transport.
A target function is allowed to be wrong; it is never allowed to differ between fill and
backward.

---

## 2. What changes, what does not

### 2.1 The seam *(rev)*

| old identifier | new identifier | file | change |
|---|---|---|---|
| `utilEmitterPick` | `emitterSelection(ξ)` | `light/emitter.slang:20` | **kept as the fill's candidate generator**; called with `ξ` remapped onto the mesh range (§2.3) |
| `utilEmitterSampleFromPick` | `emitterSample(ξ: f32v4)` | `light/sample.slang:28` | **split** into `emitterSampleFromSelection(selection, ξ.yzw)` plus a thin `emitterSample` that does selection and then calls it. The fill needs to pass its own (remapped) selection |
| `utilEmitterPickPdf` | `emitterPickPdf(lightIndex)` | `light/emitter.slang:66` | gives `pEnv` only; no longer the mesh selection density on the ReGIR path |
| `utilEmitterFindForHit` | `emitterIndexFromHit(hit)` | `light/emitter.slang:85` | **binary search** (§2.2) |
| `utilEmitterPdf` | `emitterPdf(hit, geometry)` | `light/sample.slang:133` | becomes the **fallback** density, used when the previous vertex's lookup missed (trap 22) |
| — | `emitterPdfEnvironment(dir)` | `light/sample.slang:123` | unchanged |
| `LightPick` / `utilLightPick` | `implSampleNee(transport, position, bounceSampler, stream, λ)` | `path/nee.slang:28` | gains `normal: Optional<f32v3>` (none at a medium vertex) and returns the cell `slot` in `ImplNeeConnection` |
| `previousEnvPickPdf` | `ImplPathScatterState` | `path/path-vertex.slang:~70` | gains `u32 regirSlot` |

**MIS sites in the new core: two forward and two backward, and every forward one goes
through `implSampleNee`.**

| site | file | what ReGIR changes |
|---|---|---|
| NEE at surface vertex | `implSamplePathHitEventSurface`, `sample-path.slang:~240` | passes `geometry.shadingNormal` (see §3.3 for which normal keys the cell) |
| NEE at medium scatter | `samplePath` `MediumScatter` case, `sample-path.slang:~85` | passes `none`, which selects `skRegirNormalBucketNone` |
| backward, surface emitter hit | `sample-path.slang:221-238` | reads `previousScatter.regirSlot`; `regirPdfArea` or the fallback `emitterPdf` |
| backward, escape | `sample-path.slang:138-147` | **none**: env density is `emitterPdfEnvironment`, unchanged |

`TransportEvent::Emission` (an emitter passed through) keeps weight 1, because NEE cannot
sample it. That does not change.

### 2.2 Fix this first, independently *(rev)*

`emitterIndexFromHit` (`light/emitter.slang:85-103`) linear-scans
`model.emitterLightCount`. With bistro as one model that is **~1 200 iterations per
emissive-triangle hit**, on the BSDF-side MIS path, at every bounce. Within a model's range,
entries are appended in meshlet order (`emitter_meshlet_inputs_from_scene`,
`emitter-tables.cpp:27`, keeps a running `primitiveIndexBase`), so `primitiveBaseIndex`
is strictly ascending:

```slang
public func emitterIndexFromHit(hit: vkof::SurfaceHit) -> Optional<u32> {
	let model: GpuResolveModelIndirect * = vkof::modelFromHit(hit);
	if (model.emitterLightCount == 0u) { return none; }
	let lights: vkof::SceneEmitterLight * = vkof::global.sceneEmitterLights;
	// -- last entry whose base is <= the hit's element
	var lo: u32 = model.emitterLightOffset;
	var hi: u32 = model.emitterLightOffset + model.emitterLightCount - 1u;
	while (lo < hi) {
		let mid: u32 = (lo + hi + 1u) / 2u;
		if (lights[mid].primitiveBaseIndex <= hit.element) { lo = mid; } else { hi = mid - 1u; }
	}
	let light: vkof::SceneEmitterLight = lights[lo];
	if (
		light.kind == vkof::EmitterKind.SceneEmitter
		&& hit.element >= light.primitiveBaseIndex
		&& hit.element < light.primitiveBaseIndex + light.primitiveCount
	) {
		return Optional<u32>(lo);
	}
	return none;
}
```

The upper-mid `(lo + hi + 1) / 2` is required. With lower-mid, `lo = mid` never advances
when `hi == lo + 1`. The precondition (strictly ascending per model) belongs as an
assertion in `emitter_lights_finalize`, not as an assumption in the shader. It is local
and correct regardless of everything below (P0.5).

### 2.3 The environment map does not enter the grid *(rev)*

The new core has no separate env coin. The env is **one entry of the same power CDF**
(`global.environmentEmitterIndex`), appended last. ReGIR makes the two-level split
explicit:

```
pEnv = emitterPickPdf(global.environmentEmitterIndex)   // 0 when out of range
ξ.x >= 1 - pEnv   ->  env branch, exactly as today: pdf = pEnv * environmentPdf(ω)
otherwise         ->  mesh branch via ReGIR; overall density (1 - pEnv) * p_regir(x)
fill candidates   ->  emitterSelection(ξ * (1 - pEnv)) -- never lands on the env entry
                      source density p_i / (1 - pEnv) / primitiveCount / elementArea
```

The `ξ * (1 - pEnv)` remap is only valid while the env entry is last. Assert this in
`emitter_lights_append_environment`'s caller (`scene-state.cpp:248`), or drop the
assumption and compute `meshCdfEnd = lights[envIndex - 1].cdf` explicitly.

**The env side needs no edit.** `emitterPdfEnvironment` already equals `pEnv * pdf`, and
the escape-branch MIS weight is untouched. An env map has no position and no finite area,
so it does not belong in a spatial cache, and it is already importance-sampled (alias
table, `light/environment.slang`).

**Where the `(1 - pEnv)` goes:** in the forward contribution (`W` is conditional on the
mesh branch, so divide by `1 - pEnv`) *and* in both MIS densities. Leaving it out of one
side is trap 2 from `plan-nee.md` in a new form.

### 2.4 What is not broken *(rev)*

Per §R.2, the suite that pins mean-correctness runs on `shaders-old/` and **does not
protect the new core**. Before ReGIR, P-1 builds a new-core equivalent of at least:

- tier 2's rectangle-light Lambert closed form (`test-emitter-nee-scenes.cpp:1194`);
- NEE on vs off convergence (`:872`), env + mesh linearity (`:1553`), and the env/mesh
  power-split invariance (`:1682`). The last one is the test that sees a missing
  `(1 - pEnv)`;
- NEE on vs off inside a medium (`:1890`).

The harness is a test entry that builds a `vkof::Sampler`, constructs a flat or medium
transport, runs `vkof::samplePath` with `PathConsumer_Evaluate`, and reads back `Li`. Its
include path is `PONDER_SHADER_NEW_DIR`, which `CMakeLists.txt:438` already defines for
`ponder`. Unit tests need it exported to `unit-test/CMakeLists.txt` as well. Once these
pass on the new core, **every one of them is a regression test for this work**, and a
selection strategy that moves any mean is a bug by construction.

---

## 3. Data structures

### 3.0 Where it lives *(rev)*

| piece | new home | notes |
|---|---|---|
| layouts | `types/regir.slang` (`implementing types`): `vkof::RegirCellKey`, `vkof::RegirCell`, `vkof::RegirReservoir`, `skRegir*` constants | port `util-regir-shared.h` 1:1. A C++ mirror goes in `ponder/wavefront.hpp` with `static_assert(sizeof/offsetof)`, following the `GpuGlobal` pattern at `wavefront.cpp:116` |
| grid + reservoir + select + backward pdf | `light/regir.slang` (`implementing light`, `__include` in `light.slang`) | port `util-regir.slang` and `util-regir-fill.slang`. Not `light/light.slang` (slangd quirk) |
| fill entry | `entry/wavefront-regir-fill.slang` | port `app/viewer/shaders/regir-fill.slang`, reading `vkof::global` instead of `GpuGlobalExtended` |
| buffers | owned by `ponder::Wavefront` (`wavefront_create` / `wavefront_resize`), not the viewer's scene-state | move `regir_grid_ensure` (`scene-state.cpp:295`) into `wavefront.cpp` |
| globals | `types/global.slang` + `GpuGlobal` in `wavefront.cpp` | `regirGuards: u32*`, `regirCells: RegirCell*`, `regirReservoirs: RegirReservoir*`, `regirCellCount`, `regirCellSize`, `regirEnabled`, `regirFillIndex`. Append at the end and extend the `offsetof` asserts |
| render node | `ponder::wavefront_regir_fill(...)` → `vkof::RenderNode`, ordered **before** `wavefront_integrate` | the node order is the barrier for trap 23 |

### 3.1 Reservoir

The stored sample must be a **point on a light**, not a light index — `hat_h` contains
`cos(theta_L)/d^2`, which needs a point.

```c
// lib/ponder/shaders/regir-shared.h
struct GpuRegirReservoir {
	// enough to reconstruct the sample and re-evaluate its material at shading
	u32   lightIndex;
	u32   primitiveIndex;
	f32v2 barycentric;      // (b1, b2); b0 implied, matching the ray-query convention
	// unbiased contribution weight; multiply by this, never divide by a pdf
	f32   unbiasedWeight;
	// hat_h(Y) at fill time, cached so the shading-stage resample and the backward
	// pdf do not recompute it inconsistently
	f32   targetValue;
};
```

Position and normal are *not* stored — recompute them from `(lightIndex, primitiveIndex,
barycentric)` via `utilEmitterTriangle`, exactly as the forward sampler does, so there is
one source of truth for where the sample is. Storing them would be a second one, and
`plan-nee.md` trap 4 is what happens when two sources of triangle identity disagree.

*(rev)* In the new core, "the same as the forward sampler" means rebuilding a
`vkof::SurfaceHit` and calling `surfaceGeometryFromParameters`, which is what
`emitterSample` does at `light/sample.slang:76-86`:

```slang
var hit: vkof::SurfaceHit;
hit.kind = vkof::GeometryKind.Triangle;
hit.instance = light.modelDrawIndex;
hit.element = res.primitiveIndex;       // flat, already includes primitiveBaseIndex
hit.parameters = res.parameters;        // stored verbatim from emitterSample's hit.parameters
let geometry: vkof::SurfaceGeometry = vkof::surfaceGeometryFromParameters(hit);
```

So **rename `barycentric` to `parameters` and store `hit.parameters` unconverted**. The
old `utilRegirSelect` expanded the pair to `(b0, b1, b2)`. That is a second convention, and
trap 4 again. Keep `lightIndex` alongside `primitiveIndex`; `modelDrawIndex` comes from
the entry.

### 3.2 Cell

```c
struct GpuRegirCell {
	// hash checksum for collision detection under linear probing
	u32   checksum;
	// representative point + normal the target functions were evaluated at
	f32v3 representativePosition;
	f32v3 representativeNormal;
	// (1/M) * sum(hat_h(X_i)/p(X_i)) from the fill loop -- the MC estimate of
	// integral hat_h dA. one per target function. this is what turns hat_h into a pdf
	f32   normalizationNonCanonical;
	f32   normalizationCanonical;
	// the frame index this cell was last filled; drives the refresh policy (section 6.1)
	u32   lastFillIndex;
};
```

Reservoirs live in a parallel flat array, `cellIndex * (R_N + R_C) + slot`.

### 3.3 Hash grid

Per Clabault: two independent hashes, one for the slot and one as a checksum, plus a
quantized normal in the key so cells segregate by orientation.

```c
ivec3 gridCoord = ivec3(floor(worldPosition / cellSize));
u32 checksum  = xxhash32(gridCoord.z + xxhash32(gridCoord.y + xxhash32(gridCoord.x)));
u32 cellHash  = pcgHash(gridCoord.z + pcgHash(gridCoord.y + pcgHash(gridCoord.x))) % cellCount;
```

Linear probing, cap the probe at 32 steps, target ~60% load factor, log evictions and
probe-length distribution. The normal is folded into the key by quantizing each component
to a small number of buckets.

**Sizing.** Cell size in world units is the dominant quality knob — Clabault found
0.5–1.0 units best, with finer grids helping glossy surfaces and hurting elsewhere.
bistro's scale must be measured (P0), not assumed; glTF units here are metres-ish but the
`modelMatrix` is not necessarily unit scale.

*(rev)* **Which normal keys the cell.** Use `geometry.geometricNormal`, not
`shadingNormal`. The key has to be reproducible from the vertex alone, and normal maps
would scatter one flat wall across buckets. The `cos(theta_s)` in the shade-time resample
(§4.3) uses `shadingNormal`, because that is the integrand. The backward pdf never
rehashes (trap 20), so the choice only affects variance. Make it deliberately anyway.

### 3.4 Sampler change *(rev)*

Moot: `emitterSample` already returns area density (§R.4). What remains is the
`emitterSampleFromSelection` split from §2.1. After it, the fill loop calls the same
triangle-and-point sampler as NEE, with a selection it built itself:

```slang
// light/sample.slang
public func emitterSampleFromSelection(selection: EmitterSelection, ξ: f32v3) -> EmitterSample;
public func emitterSample(ξ: f32v4) -> EmitterSample {
	let selection: EmitterSelection = emitterSelection(ξ.x);
	if (!selection.valid) { return implEmitterSampleInvalid(); }
	return emitterSampleFromSelection(selection, ξ.yzw);
}
```

Fix while splitting: `emitterSample`'s tail has a dead `light.kind == Environment` branch
(`light/sample.slang:98-108`) that cannot be reached past the early return at :47. Delete
it.

---

## 4. The passes

### 4.0 Frame order and cell lifecycle *(rev)*

```
frame N:  [regir-fill node]  fill every claimed slot whose refresh is due (§6.1)
          [integrate node]   NEE sites: lookup -> filled? use it : claim + fall back
          [accumulate node]
```

A cell is **claimed** during integrate: the first vertex to miss CAS-claims the guard
(`utilRegirInsert`) and writes the representative position and normal. It is **filled**
by the next frame's fill node. Until then, `lastFillIndex == skRegirNeverFilled`
(`0xffffffff`), and **a claimed-but-unfilled cell counts as a miss on both the forward
and the backward side**. If only one side treated it as a miss, the pair would mix two
densities (trap 22). The one-frame lag costs variance only in frame 0 after a reset, and
only in cells not seen before.

The representative is written **once, by the claiming thread**, and is never overwritten
afterwards. The old header calls it "racy by design", and that is safe only for the
claim. A representative that moves after a fill would leave `targetValue` (fill-time
`hat_h`) and `regirPdfArea` (current `hat_h`) evaluated at different points.

Scene edits and camera resets clear the grid (port `regir-clear.slang`). The guard buffer
going back to zero is what invalidates every slot.

### 4.1 Fill *(rev)*

Per cell, per reservoir, for `L` candidates:

```slang
let pEnv: f32 = vkof::emitterPickPdf(vkof::global.environmentEmitterIndex);
let meshMass: f32 = 1.0f - pEnv;
// -- per candidate
let selection: vkof::EmitterSelection = vkof::emitterSelection(ξ.x * meshMass);
let s: vkof::EmitterSample = vkof::emitterSampleFromSelection(selection, ξ.yzw);
// s.pdf is pickPdf / primitiveCount / elementArea; condition it on the mesh branch
let p: f32 = s.pdf / meshMass;
let h: f32 = implRegirTargetNonCanonical(s, cell.representativePosition, normalOrNone);
reservoirUpdate(res, s, h / p, h);       // the 1/L is folded into finalize, as before
normAccum += h / p;
```

Then `norm = normAccum / L` and `res.unbiasedWeight = res.weightSum / (L * res.targetValue)`.
The finalize is ported verbatim from `utilRegirReservoirFinalize`
(`shaders-old/util-regir.slang:297`), including the `targetValue > 0` guard. Tier 10 pins
it. **Two deliberate differences from the old fill:**

1. **Area-measure weight, not the cancelled form.** The old sampler returned solid angle,
   so `utilRegirCandidateWeight` cancelled `cos(theta_L)/d^2` to get
   `L_e * cos(theta_s) / p_omega`. The new sampler returns `p_A` directly. Compute `h / p`
   as written above. It has no grazing singularity, because `p_A` has no cosine in it.
2. **The env is excluded by remapping `ξ`, not by rejection.** The old loop drew from the
   full list and counted env picks as zero-weight candidates (still in `M`). That is
   correct, but it wastes `pEnv` of every candidate budget. With the remap, `p` must be
   divided by `meshMass`, as above. Re-run tier 10's `E[W] == area` cases with a nonzero
   `pEnv` to pin that division.

Draw all five `ξ` per candidate before any early-out. The old loop did this for a reason
(its comment at `util-regir-fill.slang:183`): A/B comparisons between target functions
stay valid when the stream does not depend on the data. Seed with `(slot, reservoir,
regirFillIndex)` (`utilRegirFillSeed`). Without the fill index, every refill redraws the
same candidates and §6.1's refresh silently does nothing.

`implRegirTargetNonCanonical` ports `utilRegirTargetNonCanonical` unchanged:
`|cos(theta_L)|` (two-sided emission), `max(cos(theta_s), 0)`, and `1` when there is no
normal.

Run this for both target functions. `R_N` non-canonical reservoirs and `R_C` canonical;
Clabault's tuned numbers are `L = 32`, `R = 64`, `S = 4` — treat as starting points, sweep
in P4.

**Which cells to fill** is §6.2 and is the hardest scheduling question in the plan.

### 4.2 Spatial reuse

Optional, +20–30% in the published numbers, and cheap to add once fill works. Resample
neighbouring cells' reservoirs into the centre cell using **constant MIS weights**:

```
w_i(x) = 1 / (sum_j indicator(p_j(x) != 0) * T)
```

Neighbours are found by offsetting `P_c` in its tangent plane by `jitter * reuseRadius`
and rehashing — not by walking grid indices, because the normal is part of the key.

Land it after §4.1 and §4.3 are green, not with them.

### 4.3 Shade

At a NEE site with shading point `p`, normal `n`:

```
pJit  = jitterTangentPlane(p, n, rng, cellSize, 0.5)     // breaks cell-boundary structure
cell  = hashLookup(pJit, n)   // fall back to unjittered p if the jittered lookup misses
if (!cell) -> fall back to the flat power CDF path        // section 7 trap 22
```

Pull `S` reservoirs (mixing canonical and non-canonical), resample them against the *real*
integrand at `p` — full BSDF `f_r`, real `cos(theta_s)`, real distance — combine with
pairwise MIS (§5.3), and connect to the survivor with one shadow ray.

The shadow ray uses the §3.7 construction from `plan-nee.md` verbatim: derive **both**
direction and distance from the offset origin. That bug cost 13.4% of shadow rays and 1.3%
of image energy; do not re-derive it.

*(rev)* In the new core the shadow ray is `transport.visibilityQuery(origin, target,
stream)`, and the offset lives inside the transport (`plan-shadow-ray-offset.md`, done).
Nothing changes there. The shape of `implSampleNee` after P3, with S = 1 (the
`utilRegirSelect` port) first and the RIS over S afterwards:

```slang
struct ImplNeeConnection {
	f32v3 ωo;
	f32 pdfEmitFromNee;          // solid angle, the approximate p_hat for mis (§5.3)
	vkof::Spectrum radiance;     // L_e * T * W_omega -- no pdf division on the regir path
	u32 regirSlot;               // skRegirSlotInvalid on a miss; written even on the env branch
};

func implSampleNee(
	inout transport: vkof::I_TraceTransport,
	position: f32v3,
	normal: Optional<f32v3>,
	bounceSampler: vkof::SamplerBounceView,
	inout stream: vkof::SampleStream,
	λ: vkof::Wavelengths,
) -> ImplNeeConnection {
	// -- 1. cell lookup FIRST and unconditionally: the backward pdf at the next vertex
	//       needs this slot even if this vertex's coin picks the env (trap 20, trap 25)
	let jitter: f32v2 = bounceSampler.nextF32v2(vkof::BounceDimension.RegirJitter);
	let slot: u32 = vkof::regirLookupOrClaim(position, normal, jitter);
	// -- 2. env coin, exactly as today's combined cdf would land
	let pEnv: f32 = vkof::emitterPickPdf(vkof::global.environmentEmitterIndex);
	let ξSelect: f32 = bounceSampler.nextF32(vkof::BounceDimension.EmitterSelect);
	if (ξSelect >= 1.0f - pEnv) { /* env branch, unchanged */ }
	// -- 3. mesh branch
	if (slot == vkof::skRegirSlotInvalid) {
		// fallback: today's power cdf over meshes; emitterSample(ξSelect, ...) already
		// lands on a mesh entry because ξSelect < 1 - pEnv. pdf = s.pdf as today
	} else {
		let pick: vkof::RegirSelection = vkof::regirSelect(slot, bounceSampler);
		// W is conditional on the mesh branch -> divide by (1 - pEnv)
		// W_omega = W_area * areaJacobian * cosθL / (1 - pEnv)
		// pdfEmitFromNee = (1 - pEnv) * regirPdfArea(slot, x) / (areaJacobian * cosθL)
	}
	connection.regirSlot = slot;
}
```

New `BounceDimension`s: `RegirJitter` (2 wide), `RegirReservoir`, and later
`RegirResample`. `Count` goes from 9 to 12 or 13. `implBounceDimensionOffset` is
`BounceStart + bounce * Count + dim` (`sample/sampler.slang:119`), so **changing `Count`
reshuffles every dimension of every bounce > 0**. The mean is unaffected, but
bit-identity across that change ends once. Do it in its own change so a diff in an image
is not misread as ReGIR.

### 4.4 Backward pdf (the BSDF-hit MIS weight)

A BSDF-sampled ray hits an emissive triangle. To weight it you need the density the light
technique *would have had* for that direction, from the previous vertex.

```
1. lookup   = utilEmitterFindForHit(...)          // binary search, section 2.2
2. cell     = the cell used at the PREVIOUS vertex   // from path state, NOT rehashed
3. h_N, h_C = hat_h_N(x), hat_h_C(x) evaluated w.r.t. that cell's P_c, N_c
4. p_area   = combine(h_N / cell.normNonCanonical, h_C / cell.normCanonical)
5. p_omega  = p_area * d^2 / cos(theta_L)
6. weight   = misBalanceWeight(bsdfPdf, p_omega)
```

*(rev)* The same thing in the new core. It replaces `sample-path.slang:221-238` and
assumes §R.3 fixes 1 and 2 have landed:

```slang
// path-vertex.slang, ImplPathScatterState
u32 regirSlot;     // skRegirSlotInvalid: the vertex fell back (or regir is off)

// sample-path.slang, surface hit
if (useMeshNee && previousScatter.neeAvailable && mtrl.model.emitterLightCount > 0u) {
	let cosθL: f32 = abs(dot(geometry.geometricNormal, -previousScatter.direction));
	if (hit.areaJacobian > 0.0f && cosθL > 0.0f) {
		let pdfArea: f32 = (
			previousScatter.regirSlot != vkof::skRegirSlotInvalid
			? vkof::regirPdfArea(previousScatter.regirSlot, hit.surface.hit, geometry)
			: vkof::emitterPdf(hit.surface.hit, geometry)
		);
		vertex.pdfEmitFromHit = pdfArea / (hit.areaJacobian * cosθL);
	}
}
```

```slang
// light/regir.slang -- (1 - pEnv) * hat_h(x) / Norm, against the stored cell
public func regirPdfArea(
	slot: u32, hit: vkof::SurfaceHit, geometry: vkof::SurfaceGeometry,
) -> f32 {
	let optLight: Optional<u32> = vkof::emitterIndexFromHit(hit);
	if (!optLight.hasValue) { return 0.0f; }
	let cell: vkof::RegirCell = vkof::global.regirCells[slot];
	let light: vkof::SceneEmitterLight = vkof::global.sceneEmitterLights[optLight.value];
	let h: f32 = implRegirTarget(light, geometry.position, geometry.geometricNormal, cell);
	if (!(cell.normalizationNonCanonical > 0.0f)) { return 0.0f; }
	let pEnv: f32 = vkof::emitterPickPdf(vkof::global.environmentEmitterIndex);
	return (1.0f - pEnv) * h / cell.normalizationNonCanonical;
}
```

The forward side's `pdfEmitFromNee` must call **this same function** on the sampled
point. Do not inline a second copy of the formula into `implSampleNee`. The tier-11.2
round-trip test needs exactly one implementation to compare against itself.

`previousScatter.regirSlot` is written at both NEE sites (surface and medium), and reset
to `skRegirSlotInvalid` wherever `neeAvailable = false` is set. A stale slot from two
vertices back is trap 20 by another route. The camera's initial state (sample-path.slang
`previousScatter` init) sets it invalid as well.

**The refill hazard on the backward side.** The slot is read in the *next* bounce of the
same integrate dispatch, and the fill only runs between dispatches (§4.0), so the cell
cannot change between the forward and the backward read. This holds only while the fill
is a separate node. If the fill is ever merged into integrate, it breaks (trap 23).

**Step 2 is a correctness requirement, not an optimization.** The lookup at shading time is
*jittered*, so the cell is not a deterministic function of the previous vertex's position
and normal — rehashing gives a different cell with high probability, and the two halves of
the MIS weight then price the same direction against different densities. The cell index
must be threaded in path state. This is `previousEnvPickPdf`'s direct successor and
`plan-nee.md` §9 is the template for getting it right at all four sites.

### 4.5 Refresh

See §6.1. This pass does not exist in either source and is the one ponder cannot skip.

---

## 5. The math to get exactly right

### 5.1 `W`, not `1/pdf`

Every contribution is `integrand * W_Y`. There is no division by a light pdf anywhere in
the ReGIR path. Reviewing for a stray `/ pdf` is a good self-check.

### 5.2 `W_Y` is unbiased only in expectation over grid realizations

This is the sentence to tape to the monitor. `W_Y` estimates `1/p_true(Y)` where `p_true`
is the *continuous* density that the fill pass's randomness induces. Averaging many shading
draws against a **fixed** grid does not average over that randomness. See §6.1 — this is
ponder's headline problem.

### 5.3 MIS weights may use approximate densities; they may not fail to sum to 1

Worth being precise, because it changes how much of this is risky.

The balance heuristic `w_i = q_i / sum_j(q_j)` sums to 1 over `i` for **any** positive
`q`, true densities or not. The estimator `sum_i w_i(x) * f(x) / p_i(x)` is unbiased as
long as (a) the weights sum to 1 pointwise, and (b) each technique divides by its own
*true* density — or, for the light technique, multiplies by an unbiased `W`.

So substituting `hat_h / Norm` for the true ReGIR density in the MIS weight costs
**variance, not bias**. What would cost bias:

- weights that do not sum to 1 pointwise, i.e. the forward and backward halves using
  different `q` for the same direction (→ §4.4 step 2, and store `Norm` rather than
  recomputing it, so both halves read the same number);
- a direction with nonzero contribution that *neither* technique can generate (→ canonical
  reservoirs, §1.3);
- a candidate source pdf `p` that is zero where `hat_h` is nonzero (→ the power CDF is
  positive on every listed emitter, so this holds as long as the fill draws from the same
  list the backward lookup searches).

I flagged "approximate MIS weights" earlier as a first-of-its-kind bias risk for this
codebase. That was wrong; the above is the accurate decomposition, and it means the
dangerous items are §6.1 and trap 21, not the weights.

### 5.4 Pairwise MIS across canonical and non-canonical

With `|C|` canonical techniques among `M` total:

```
i not in C:   m_i(x) = 1/(M-|C|) * p_i(x) / ( p_i(x) + sum_{k in C} p_k(x)/(M-|C|) )
k in C:       m_k(x) = 1/(M-|C|) * ( p_k(x)/(M-|C|) ) / ( p_i(x) + sum_{k' in C} p_{k'}(x)/(M-|C|) )
```

Unit-testable in isolation on the CPU with synthetic densities, and I will do exactly that
(test 9.6) before it ever meets a triangle — this is the single most error-prone formula in
the plan and it has a checkable invariant (sums to 1).

---

## 6. ponder-specific problems the papers do not have

### 6.1 A converging renderer must refresh the grid — this is the headline

Ponder accumulates to tens of thousands of spp (the bistro screenshots are 8 k and 38 k).
Both sources target renderers that refill the grid every frame and never accumulate more
than a handful of samples against one grid realization.

Hold the grid fixed and the reachable sample set is finite: `R` reservoirs per cell times
the handful of cells jitter can reach — order 500 distinct light points. Averaging shading
draws against that converges to `integral f dp_hat` over a *discrete* distribution, not to
`integral f dA`. Per §5.2 the unbiasedness of `W_Y` lives in the fill randomness, and a
fixed grid never re-rolls it.

**So a stale grid is not a variance cost here. It is bias, and it does not decay with spp.**
This is the exact opposite of the stale-emitter-list contract in `plan-nee.md` §3.2, where
a stale list only changed sampling density. Do not carry that intuition over.

Options:

| policy | correctness | cost |
|---|---|---|
| refill every frame (both sources) | correct | full grid fill per frame, amortized over one tile's pixels (§6.2) |
| refill `1/K` of cells per frame, round-robin | correct; residual decays with a correlation length of `K` frames | `1/K` the cost |
| refill on demand, cells touched last frame | correct; adapts to the tile pattern | bookkeeping, and a cold-cell stall |
| never refill | **biased, plateaus** | free |

Recommended: **round-robin `1/K`**, with `K` exposed and swept, because it degrades
smoothly and makes the correlation length an explicit measurable number rather than an
emergent property.

**The test that catches this is not a standard one** and it is why I am writing tier 11:
converge to very high spp with refresh disabled and watch the error *plateau* while spp
keeps climbing. A mean-equality test at 256 spp will pass happily on a fixed grid.

### 6.2 Tile dispatch fights the fill pass *(rev: resolved)*

The old viewer dispatched a tile budget. The wavefront core dispatches the full frame
(`wavefront_integrate`, `16×16` over `width × height`), so each fill is amortized over
every pixel's path vertices. What remains of this section is one measurement in P0:
**fill ms vs integrate ms at 1 spp per frame**. If the full fill costs more than about 20%
of integrate, start §6.1's `K` at that ratio rather than at 1.

### 6.3 A vdb medium scatter has no normal

The hash key includes a quantized normal. A medium scatter vertex has none, and the phase
function is not a BSDF. Options: a reserved "no normal" bucket in the key (cells then
aggregate over all orientations, which is right for an isotropic phase function), or key
medium lookups on position only into a separate grid.

Recommended: a reserved bucket, so there is one grid. But note `hat_h_N`'s `cos(theta)`
term must degrade to 1 there, not to a garbage dot product with a zero vector — that is a
silent zero, and a silent zero in a target function is the §7 trap 21 failure.

`plan-nee.md` P4 landed mesh NEE at the medium site and found four bugs doing it; three
were about which vertex's state was in scope. Expect the same class here.

*(rev)* The reserved bucket already exists: `skRegirNormalBucketNone`, ported in §3.0.
In the new core the medium site is the `MediumScatter` case of `samplePath`, which calls
`implSampleNee(..., normal = none, ...)`, and its backward side is the same surface-hit
code, because the *next* hit is still a surface. Fix §R.3 item 3 (the `envNeeEnabled`
gate) first, or mesh NEE at medium vertices never runs and P6 tests nothing.

### 6.4 The subsurface exit vertex

Same shape as the medium site: it is where the *next* segment starts, so it needs its own
cell, not the entry vertex's. `plan-nee.md` §9 records exactly this bug on the env side.
It will recur; the fix is the same.

*(rev)* The new core has no subsurface yet (§R.4). This waits for track A.

### 6.5 Delta bounces

Unchanged from `plan-nee.md` trap 1 and still mandatory: a delta bounce (thin-walled
transmission, the TIR fallback drawing `wo = -wi`, roughness exactly 0) has no density the
light technique can match, so the backward weight must be 1, not a blend. The condition
lives in the same place; ReGIR does not change it.

*(rev)* In the new core that place is `previousScatter.neeAvailable = false`. A delta
lobe must also set `regirSlot = skRegirSlotInvalid` (§4.4). There are no delta lobes until
track A lands (§R.4).

---

## 7. Traps

Carried from `plan-nee.md` §4 and still live: 1 (delta bounces), 2 (selection density in
*every* MIS weight), 3 (`previousVertexSampledLights`), 4 (flat vs meshlet-local
numbering), 5 (sidedness on both halves), 6 (material index 0 reserved), 12 (zero emitters
must cost nothing). New:

19. **Measure mixing.** Area vs solid angle, §1.1. A single missed `d^2/cos` reads as a
    plausible image with a distance-dependent brightness error — the hardest kind to spot
    by eye and invisible to any NaN probe.

20. **The backward pdf must use the previous vertex's *stored* cell.** §4.4 step 2.
    Rehashing is wrong because the shading lookup is jittered.

21. **A silent zero in `hat_h` deletes a light.** If `hat_h_N` is zero for an emitter in
    every cell and the canonical reservoirs do not cover it, that light disappears from the
    image with no error, no NaN, and no failing mean test unless the test happens to
    include it. Canonical reservoirs exist for exactly this. Test 9.7 is the guard.

22. **The grid-miss fallback must exist and must be unbiased.** A shading point whose cell
    is absent (cold, evicted, probe-capped) must fall back to the flat power CDF path — and
    the MIS weight for that vertex must then use *that* density, not ReGIR's. Two selection
    strategies in one renderer means two densities, and the wrong one is a silent bias.

23. **Fill and shade must read the same grid.** A refresh landing mid-frame prices the
    forward and backward halves of one MIS weight against different reservoir contents.
    Double-buffer or fence.

24. **`Norm` is a noisy estimate and that is fine — as long as it is the *same* noisy
    estimate everywhere.** Store it; never recompute it at shading or in the backward pdf.

25. **Jitter changes the rng stream.** `plan-nee.md`'s bit-identical guarantees for
    mesh-only and env-only scenes were load-bearing during P3 debugging. Draw jitter
    unconditionally or accept that those guarantees end here — decide deliberately and
    write it down.

26. **Correlation from spatial reuse is not free.** Neighbouring shading points resampling
    the same neighbour cells produce correlated estimates. Harmless in a real-time frame;
    in a 38 k-spp accumulation it is another way to converge to the wrong answer if the
    reuse pattern is fixed. Re-randomize the neighbour offsets per frame.

---

## 8. Phases

*(rev)* Each row is sized as one reviewable change. "Port" means bringing `shaders-old/`
code into the new modules unchanged except for identifiers, then retargeting the
existing test at the new include path.

| phase | content | gate |
|---|---|---|
| **P-1a** | §R.3 fixes: `pdfEmitFromHit` field, geometric normal on the backward cosine, medium NEE gate | the P-1b tests below, written first and seen failing on fix 1 |
| **P-1b** | new-core harness (§2.4): `samplePath` + `PathConsumer_Evaluate` test entry; port the tier-2 closed form, on/off, env+mesh linearity, power-split invariance, medium on/off | all green on the new core |
| **P0** | measure only: get bistro (not in `assets/`); emissive entry count after `emitter-tables.cpp`; world extent → `cellSize`; `emitterIndexFromHit` scan length; RMSE-vs-time baseline on the **new core** with the power CDF | baselines recorded; §0's claim reproduced or refuted |
| **P0.5** | `emitterIndexFromHit` → binary search (§2.2); ascending-base assertion in `emitter_lights_finalize` | P-1b unchanged; scan gone from the profile |
| **P0.6** | split `emitterSample` → `emitterSampleFromSelection` (§3.4); delete the dead env branch | P-1b bit-identical (a pure refactor) |
| **P0.7** | add `BounceDimension.RegirJitter/RegirReservoir`, bump `Count`, draw nothing from them yet | P-1b means unchanged; **bit-identity knowingly broken here, once** (§4.3) |
| **P1** | port the grid: `types/regir.slang`, `light/regir.slang` grid half, buffers in `ponder::Wavefront`, `GpuGlobal` fields + asserts, clear node | tier 9 (`test-regir.cpp`) retargeted at the new module and green |
| **P2** | port the fill: `entry/wavefront-regir-fill.slang` + render node before integrate; `ξ` remap with `meshMass`; area-measure weight (§4.1) | tier 10 (`test-regir-fill.cpp`) retargeted, plus nonzero-`pEnv` cases |
| **P3a** | claim-on-miss at the **surface** NEE site; no shading use yet (every lookup still falls back) | P-1b unchanged; claimed-cell count visible in the UI |
| **P3b** | shade with S = 1 (port `utilRegirSelect`), `regirSlot` in path state, `regirPdfArea` on both sides, `(1 - pEnv)` on both sides; refresh `K` (§6.1) | tier 11; **P-1b green with ReGIR on** |
| **P4** | measure and sweep: `L`, `R`, `S`, `cellSize`, `K`. This is where the feature either pays or does not | tier 12 |
| **P5** | canonical reservoirs + pairwise MIS | tier 13; trap 21 closed |
| **P6** | medium NEE site (§6.3). Subsurface exit and delta lobes wait for track A | medium on/off (P-1b) green with ReGIR on |
| **P7** | spatial reuse (§4.2) | tier 12 re-run |
| **P8** | visibility term in `hat_h_N` (NEE++ style, fed by shadow rays already traced) | tier 12 re-run — expected to be the largest single win on bistro |

The P3 gate (from the roadmap) is P3b vs P0's baseline at equal time.

P3 is the real gate. If the surface-vertex-only, non-canonical, no-reuse version does not
beat the flat CDF on bistro at equal time, the tuning in P4 is unlikely to rescue it and
the honest move is to stop and reconsider — that is what P0's baseline exists to make
decidable.

---

## 9. Tests — mine

Conventions per `plan-nee.md` §5: doctest in `unit-test/src/`, compute shaders in
`unit-test/shaders/`, chi-square on the `util-sampling-bins.glsl` 40x80 grid with the
Pearson + Wilson-Hilferty convention, heavy render sweeps behind `--thorough` per the GPU
cost budget. New file: `test-regir.cpp`. All of `plan-nee.md` tier 0-3 runs unchanged as
regression.

*(rev)* Tier 9 (`test-regir.cpp`) and tier 10 (`test-regir-fill.cpp`) exist and pass
against `shaders-old/`. P1 and P2 retarget them; they are not rewritten. "`plan-nee.md`
tier 0-3 as regression" now means the P-1b new-core ports (§2.4). The old-pipeline
originals keep running until `shaders-old/` is deleted, and they test nothing ReGIR
touches.

### Tier 9 — hash grid (P1)

| id | test | catches |
|---|---|---|
| 9.1 | insert/lookup round-trip over a swept point cloud; checksum rejects a colliding-but-different cell | the probing logic |
| 9.2 | load factor, probe-length distribution and eviction count under 1x / 2x / 4x capacity pressure — recorded, and mean-invariance asserted downstream | graceful degradation, trap 22 |
| 9.3 | normal quantization: points at the same position with opposing normals land in different cells; the reserved normal-free bucket is reachable and distinct | §6.3 |
| 9.4 | coordinate edge cases: negative coordinates, exact cell boundaries, positions at 1e-6 and 1e6, denormals | `floor` on negatives is the classic bug here |

### Tier 10 — RIS and fill (P2)

| id | test | catches |
|---|---|---|
| 10.1 | **`E[W_Y] == 1/p_true(Y)`** over 10^7 fill realizations against an analytically integrable single-triangle configuration | the core unbiasedness claim of §5.1, directly |
| 10.2 | `Norm` converges to `integral hat_h dA` computed by CPU quadrature, for 3 target-function variants | §1.2's free-normalization claim |
| 10.3 | reservoir survivor distribution vs `hat_h`-proportional, chi-square | the WRS update |
| 10.4 | `L = 1` degenerates to plain source sampling; `hat_h == p` degenerates to uniform survivor choice | two analytic special cases that bracket the implementation |
| 10.5 | degenerate: all candidates zero `hat_h`, one candidate, zero-area triangle, `p` denormal | the fill loop's guards |

### Tier 11 — estimator and the refresh policy (P3)

| id | test | catches |
|---|---|---|
| 11.1 | ReGIR-on vs ReGIR-off **mean equality** on every tier-2 analytic scene from `plan-nee.md` | the headline invariant |
| 11.2 | **round-trip**: forward selection density == backward `utilRegirPdf` for the same `(cell, x)`, ~1e-6 rel | trap 20, trap 24 — the property MIS rests on |
| 11.3 | **the refresh test (§6.1)**: converge to 64 k spp with refresh disabled; assert the error *plateaus* (this test asserts the bug exists). Then with refresh at `K = 1, 4, 16`, assert error keeps falling as `1/sqrt(N)` through 64 k spp | the failure neither source has to worry about and no 256-spp test can see |
| 11.4 | grid-miss fallback: shrink capacity until most lookups miss; mean must not move | trap 22 |
| 11.5 | mid-frame refresh with forward and backward reading different grids — constructed deliberately, asserted to fail, then asserted fixed | trap 23 |
| 11.6 | jitter on/off: mean invariant; rng-stream guarantees documented either way | trap 25 |

### Tier 12 — variance, which is the point (P4, P7, P8)

Gate on **RMSE vs a high-spp reference at equal spp *and* equal time**. Behind `--thorough`.

| id | test | catches |
|---|---|---|
| 12.1 | 1 000 small scattered emitters in a room: ReGIR vs flat CDF | the synthetic best case |
| 12.2 | **bistro, night, interior viewpoint**: flat CDF vs ReGIR, equal-spp and equal-time | the real target |
| 12.3 | one dominant emitter + 1 000 negligible: ReGIR must not be *worse* | the regression a bad `hat_h` produces |
| 12.4 | emitters behind an occluder: measured before and after P8 | isolates the visibility term's contribution |
| 12.5 | parameter sweep `L`/`R`/`S`/`cellSize`/`K`, recorded as a table | P4's actual deliverable |
| 12.6 | cost accounting: fill ms, shade ms, shadow rays, probe length, ns/query | equal-time honesty |

### Tier 13 — canonical reservoirs (P5)

| id | test | catches |
|---|---|---|
| 13.1 | pairwise MIS weights sum to 1 pointwise, on synthetic densities, CPU-only, before any triangle is involved | §5.4 — the most error-prone formula in the plan |
| 13.2 | an emitter for which `hat_h_N` is identically zero in every cell still renders at the correct mean | **trap 21** |
| 13.3 | canonical-only and non-canonical-only both hit the same mean as the combination | that the combination is a partition, not a blend |

---

## 10. Decisions

1. **Refresh policy** (§6.1): round-robin `1/K` (recommended) vs per-frame full fill vs
   on-demand. Sets the correlation length and interacts with tile dispatch (§6.2).
   **This is the decision the plan hinges on.**
2. **Reservoir stores a point sample** (recommended, and effectively forced by `hat_h`
   containing `cos(theta_L)/d^2`) vs a light index with the point re-drawn at shading —
   cheaper storage, but then `hat_h` cannot see the geometry term and most of the win goes.
3. **Medium vertices**: reserved normal bucket in one grid (recommended) vs a second
   position-only grid.
4. **Jitter and the rng stream** (trap 25): keep `plan-nee.md`'s bit-identical guarantees
   by drawing unconditionally, or let them go.
5. **Light BVH as candidate generator, later or never.** Not needed now; the power CDF is
   the RTG2 baseline. Revisit only if P4 shows candidate quality is the binding constraint
   — and note §0's evidence that on bistro it may not help at all.
6. **Whether the observer wants any of this.** It is a visibility-buffer rasterizer whose
   resolve does prefiltered split-sum IBL only — it has a TLAS bound but traces nothing and
   has no emissive or shadow path. So: nothing to integrate with today. Recorded so the
   answer is on paper rather than assumed.

---

## 11. Relationship to `plan-nee.md`

**Carried forward unchanged:** emitter identity as `(modelDrawIndex, primitiveIndex)` in
the flat numbering; `GpuEmitterLight` / `vkof::SceneEmitterLight` as the record;
two-sided emission; radiance evaluated at the sampled point's material; the delta-bounce
exclusion; the env branch; the shadow-ray origin construction from §3.7.

*(rev)* **Not carried into the new core:** the hybrid spherical/area sampler, because
`emitterSample` is area-uniform (§R.4). There are now two forward sites and two backward
sites, not four (§2.1). `previousEnvPickPdf` is replaced by `emitterPdfEnvironment`. The
tier 0-3 suite becomes regression only once it is ported (P-1b).

**Superseded:** §3.2's power-CDF selection *as the shading-time strategy* — it survives as
the fill pass's candidate generator. §3.6 named a light BVH as the successor; the evidence
in §0 says ReGIR instead, and §3.6's claim that `emitterSample`/`emitterPdf` is the right
seam turned out to be correct regardless.

**Reversed:** §3.2's "a stale or capped table changes only the sampling density" does
**not** generalize to a stale grid. §6.1. That intuition is now actively dangerous.

**Still open there and untouched here:** `plan-nee.md` P5 (golden images, logging, README)
and tiers 3.6-3.13, of which 3.7-3.13 are unwritten — including 3.13, the delta-lobe sweep
for the "thin glass renders too dark" bug, which still has no coverage. Orthogonal to
selection; own schedule.

---

## 12. Sources

- [Boksansky, Wyman et al., *Rendering Many Lights with Grid-Based Reservoirs*, Ray Tracing Gems II ch. 23](https://cwyman.org/papers/rtg2-manyLightReGIR.pdf) — the original ReGIR
- [Tom Clabault, *ReGIR — an advanced implementation for many-lights offline rendering* (2025)](https://tomclabault.github.io/blog/2025/regir/) — the offline adaptation this plan follows, with the target-function ablation, the canonical/non-canonical scheme and the bistro numbers
- [TomClabault/HIPRT-Path-Tracer](https://github.com/TomClabault/HIPRT-Path-Tracer) — that renderer's source, inspectable
- [Bitterli et al., *Spatiotemporal reservoir resampling…* (ReSTIR), SIGGRAPH 2020](https://dl.acm.org/doi/10.1145/3386569.3392481)
- [Lin et al., *Generalized Resampled Importance Sampling*, SIGGRAPH 2022](https://research.nvidia.com/labs/rtr/tag/restir/) — the framework the `m_i` weights come from
- [Wyman et al., *A Gentle Introduction to ReSTIR*, SIGGRAPH 2023 course](https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf)
- [Kettunen et al., *Conditional Resampled Importance Sampling and ReSTIR*, SIGGRAPH Asia 2023](https://dl.acm.org/doi/10.1145/3610548.3618245)
- [Guo, Eisemann & Eisemann, *Next Event Estimation++*, CGF 2020](https://graphics.tu-bs.de/publications/guo2020next) — the visibility term for P8
- [NVIDIA RTXDI](https://github.com/NVIDIA-RTX/RTXDI) — ships ReGIR as its world-space light sampler
- [Conty Estevez & Kulla, *Importance Sampling of Many Lights with Adaptive Tree Splitting*, HPG 2018](https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf) — the light-tree alternative, for §10.5
