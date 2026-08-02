# NEE for emissive meshes — implementation plan

Scope: next-event estimation toward **emissive triangles only**, MIS-combined with
BSDF sampling and with the existing env-map NEE. Analytic lights (directional, quad,
point) are explicitly *not* in scope; the interfaces below are shaped so a second
emitter kind slots in as another entry in the same light list, but nothing analytic gets
written.

Revision 2 — reworked after reading `~/repo/cull`'s deleted implementation and
cross-checking Cycles and PBRT-v4. See §8 for what changed and why.

---

## 1. What already exists

| piece | where | reuse |
|---|---|---|
| **a complete prior implementation** | `~/repo/cull/app/shaders/util-lights.glsl`, `app/shared/light_shared.h`, `app/main.cpp:1075-1215` | **port, don't reinvent** — see §8.1 |
| env NEE at a surface vertex | `app/viewer/shaders/util-raytrace.glsl:310-406` | structural template |
| env NEE at a subsurface exit vertex | `util-raytrace.glsl:883-900` | second integration site |
| env NEE at a vdb medium vertex | `util-raytrace.glsl:1292-1330` | third integration site |
| BSDF-side MIS reweight on env escape | `util-raytrace.glsl:1505-1530` | fourth site, needs a mixture-pdf edit |
| balance heuristic | `lib/ponder/shaders/util-mis.glsl` | as-is |
| Vose alias builder | `lib/ponder/src/environment-tables.cpp:38-88` (file-static) | extract and share |
| **per-meshlet object-space area** | `mor::scene_meshlet_area_info`, `mor.hpp:78-86` | this API exists *because* cull's emitter build needed it |
| emission on the material | `GpuMorMaterial.emissionLuminance` / `.emissionColor` | as-is |
| glTF emissive load | `lib/mor/src/mor.cpp:1197-1208`, `:1655-1694` | as-is |
| `envNeeEnabled` uniform + toggle | `main.cpp:854`, `:1054`, `:2397` | copy for `meshNeeEnabled` |

Emission is currently added **raw at every path vertex** (`util-raytrace.glsl:252-257`),
with no MIS weight. That line is the BSDF-side half of the estimator and is the most
delicate edit in this plan.

## 2. Prerequisites — two test harnesses are currently dark

- **`unit-test/src/test-dielectric-scenes.cpp` is not in `unit-test/CMakeLists.txt`.**
  The only file in `unit-test/src/` missing from the explicit source list. It carries
  `cornell_wall_records()` / `cornell_meshes_with_sphere()` / `skCornellCamera` — the
  cornell harness the headline test needs. `test-cornell.cpp` no longer exists; its
  helpers live here. The cornell renders in the README are currently not reproducible
  from the suite.
- **`unit-test/src/test-furnace-model-render.cpp` is `#if 0`'d** at line 1-2 — the real-glTF
  NaN/firefly probe tier 3.11 wants.

Re-enabling either may surface pre-existing failures unrelated to NEE; treat that as a
separate finding rather than folding it into this work.

---

## 3. Design

### 3.1 Emitter identity

A traced hit resolves as `(instanceCustomIndex, primitiveIndex, barycentric)`
(`util-raytrace.glsl:79-83`), and `instanceCustomIndex == modelDrawIndex ==` the index
into the `GpuResolveModelIndirect` array (`app/viewer/scene-state.cpp:238`). That entry
carries `modelMatrix`, `positions`, `flatIndices`, `flatMeshlets`, `meshlets`,
`materials`, `attributes`.

So an emitter is a **`(modelDrawIndex, primitiveIndex)` pair** in the *flat* primitive
numbering — the same numbering ray hits report, **not** the meshlet-local
`triangleOffset`/`meshletTris` numbering the raster path uses. cull's header calls this
out explicitly; getting it wrong makes the forward sampler and the backward hit lookup
disagree about which triangle is which, which reads as a plausible-looking but
systematically wrong image.

`utilMeshAttributeDataFromIndices` does not fill world positions (its `origin` field is
declared but never written). Port cull's `fnEmitterTriangleWorldPositions` /
`fnEmitterTriangleGeometry` into `lib/ponder/shaders/util-mesh-raytrace.glsl`.

### 3.2 CPU-side emitter table — `lib/ponder/{include/ponder,src}/emitter-tables.{hpp,cpp}`

**One record per emissive meshlet**, not per triangle (revised — see §8.1):

```cpp
struct GpuEmitterLight {
	u32 modelDrawIndex;
	// flat primitiveIndex of this emitter's first triangle; triangles
	// [primitiveIndexBase, primitiveIndexBase + triangleCount) are this emitter
	u32 primitiveIndexBase;
	u32 triangleCount;
	// world area * luma(emissionColor) * emissionLuminance
	f32 power;
	// cumulative power / totalPower; monotonic, last entry is 1.0
	f32 cdf;
	// emissionColor * emissionLuminance, precomputed
	f32v3 radiance;
};
```

`GpuResolveModelIndirect` grows `emitterLightOffset` / `emitterLightCount` so the
backward lookup only scans that model's own emitters.

Why per-meshlet: meshlet triangles are contiguous in the flat numbering, so one record
covers `triangleCount` triangles with a base+count range; the table stays small enough
to **rebuild every frame** (which is what keeps live material edits correct with zero
invalidation machinery); the backward lookup is a range test rather than a search; and
`mor::scene_meshlet_area_info` already computes the object-space area per meshlet, which
only needs the instance transform's scale applied. The cost is uniform triangle
selection within a meshlet (§3.3) — unbiased, but higher variance when one meshlet mixes
wildly different triangle areas. Test 1.6 stresses exactly that.

Selection: power-weighted CDF over the whole list, binary search. **Emissive textures are
ignored in `power` and in `radiance`** for v1 — v1 uses the material factors only, so a
textured emitter is sampled (and, in cull's design, *shaded*) with the average rather
than the texel value. This is where ponder should diverge from cull: evaluate the
sampled point's radiance through `openPbrLoadMaterialDeriv` at the sampled uv, exactly
as the BSDF-hit branch does, so the two techniques cannot disagree. Keep the factor-only
approximation in `power` (steering only, never radiance).

Build: rebuild every frame alongside the models-indirect upload, with a
`kMaxEmitterLights` cap and a log when it clips. Builder takes `mor::Scene` + transform +
`modelDrawIndex`, so no app types leak into `lib/`.

**The table carries no geometry.** A stale or capped table changes only the sampling
density; a triangle absent from it gets `pdf == 0`, so `misBalanceWeight(bsdfPdf, 0) == 1`
and the BSDF technique carries it at full weight. Unbiased either way — assert it (t3.10).

### 3.3 GPU sampler — `lib/ponder/shaders/util-emitter.glsl`

Port cull's `fnPickEmitterLight` / `fnSampleEmitterTriangle` / `fnEmitSolidAnglePdf` /
`fnFindEmitterLightForHit` essentially verbatim; they are correct and already match
ponder's buffer layout.

- **Emitter pick**: power CDF binary search → `pickPdf`.
- **Triangle pick within the emitter**: uniform, `pdf_area = 1 / (triangleCount · area_tri)`.
- **Point in triangle**: `su0 = sqrt(ξ.x); b0 = 1 - su0; b1 = ξ.y·su0`.
- **Area → solid angle**: `pdf_ω = pickPdf · pdf_area · d² / cos θ_l`.

**Sampling scheme, revised.** Both Cycles and PBRT-v4 use *spherical triangle*
(solid-angle) sampling, falling back to area sampling only outside a stability window:

- PBRT-v4: spherical when the subtended solid angle is in `[3e-4, 6.22]` sr, uniform area
  otherwise; with a reference normal it additionally warps by a bilinear fit of the
  cosine at the four spherical-triangle corners (clamped to 0.01).
- Cycles: spherical when `longestEdge² > distanceToPlane²`, area otherwise; solid angle
  via Van Oosterom–Strackee, `Ω = 2·atan2(mixedProduct, 1 + cos a + cos b + cos c)`;
  `pdf = 1/Ω`; `Ω == 0` returns pdf 0.

This matters here specifically: **a cornell ceiling panel viewed from the floor sits
squarely in the regime where area sampling is the bad choice** — which is the headline
test scene (3.1–3.3) and the variance target (3.2). Worth noting that uniform spherical
sampling gives a *constant* `pdf = 1/Ω`, so the backward MIS lookup gets simpler, not
harder: no `d²/cos` Jacobian at all.

Plan: land area sampling first (P1, port of cull, simplest thing that passes tier 1),
then add the hybrid behind the *same* `emitterSample`/`emitterPdf` interface as P1.5.
The round-trip test (1.1) and chi-square (1.2) are precisely the harness that makes
swapping the sampler underneath safe, and test 1.7 asserts the two schemes agree in
distribution.

**Sidedness.** cull is one-sided (`cosAtLight > 0` required, both directions). Cycles
makes it a per-material setting — front / back / both / none — via `SD_MIS_FRONT` /
`SD_MIS_BACK`, and *rejects the sample* when the sampled side lacks the flag. ponder's
BSDF-hit branch today is two-sided (`util-raytrace.glsl:252-257` has no facing test).

Whatever is chosen, **both halves of the estimator must agree**, and this is a live bug
class even in mature renderers: Blender #115997 is "certain emission sampling settings
render differently with light tree on vs off". Recommendation: two-sided, since it
matches ponder's current behavior and glTF's double-sided default, and add the
per-material front/back/both/none knob later if it's ever wanted. Test 3.7 pins it.

### 3.4 MIS integration — four sites

**(a) Light-sampling side, surface vertex** (`util-raytrace.glsl:319`). Shadow ray with
`maxDist = distance - 0.001`, origin offset along the geometric normal, gated on
`dot(itFrame.nor, wo) > 0` and (if one-sided) `cosAtLight > 0`. Reuses the existing
`utilTraceRay` shadow call and `utilVdbFogTransmittance`. Note the ordering constraint:
NEE must use the throughput *before* this bounce's f·cos/pdf update — cull carries an
explicit `irradianceThroughputPrevious`; ponder's env block gets it for free by running
before the update, and the mesh block must stay in that same position.

**(b) BSDF side** (`util-raytrace.glsl:252-257`):

```glsl
f32 weightB = 1.0f;
if (meshNeeActive && !previousBounceWasDelta && previousVertexSampledLights) {
	const f32 lightPdf = emitterPdf(/* hit model/prim, dist, dir, geomN */);
	weightB = misBalanceWeight(bsdfPdf, lightPdf);
}
irradianceAccumulator += emission * irradianceThroughput * weightB;
```

**Delta-bounce exclusion is not optional.** Both references implement it and cull hit it
concretely: thin-walled transmission samples the exact direction `wo = -wi`, which NEE
has zero probability of reproducing, so blending against a nonzero `lightPdf`
"systematically discounts direct light seen straight through thin glass". Cycles carries
`mis_ray_pdf` + `min_ray_pdf` in path state and skips MIS when `min_ray_pdf == 0` (a
delta bounce).

In ponder the delta-ish cases are: thin-walled transmission (`sampledTransmission &&
geometryThinWalled > 0`), `openPbrSampleWo`'s TIR fallback (which draws `wo == -wi`
exactly, see the `bsdfPdf <= 0` guard at `util-raytrace.glsl:503-513`), and roughness
exactly 0. Test 3.13 covers all three.

`previousVertexSampledLights` is the companion flag: NEE is deliberately skipped at
vertices inside a closed mesh (`util-raytrace.glsl:1123-1125`), and at `i == 0` there is
no preceding BSDF sample at all. A path arriving at an emitter from such a vertex was
not reachable by the light technique; weighting it down loses energy.

**Structural note.** cull computed this MIS weight *inside the same vertex function that
traced the ray*, so no path state was needed. ponder split the trace into
`utilIrradianceWalk`, so emission accumulates one call later — hence the threading of
`hitModelDrawIndex`, `hitPrimitiveIndex` and hit distance (all three already on
`RayQueryResult`, `util-raytrace.glsl:126-137`) plus the two flags above. This is the
single biggest delta from the prior implementation and the most likely place to
introduce a bug that only shows up as a few-percent brightness error.

**(c) subsurface exit vertex** (`:889`) and **(d) vdb medium vertex** (`:1295`): identical
light-side block, with the exit-refraction pdf and the phase pdf standing in for
`bsdfPdf`, exactly as the env case already does via `outExitPdf` / `phasePdf`. Neither
existed in cull — no prior art, so these get the most test attention (3.5).

### 3.5 Mixing with env NEE — one list, power-weighted

Revised (§8.1): rather than a fixed 0.5/0.5 technique split, fold the environment map in
as **one more entry in the same power-weighted light list**, with `power` = its total
luminance integral (the `totalSum` `environment_tables_create` already computes). Pick
one entry per vertex, sample within it, one shadow ray.

This is what both references do — cull appended the sun and sphere lights to the mesh
emitter list as `triangleCount == 0` sentinel entries, and Cycles puts lamps, mesh
lights and the background in one distribution. It gives power-proportional sampling for
free, makes a future analytic light one more sentinel kind, and keeps per-vertex cost
identical to today.

The combined density is then `pdfLight(ω) = Σ_entries pickPdf_e · pdf_e(ω)`, in practice
`pickPdf_env · pdfEnv(ω) + pickPdf_tri · pdfTri(ω)` since a given direction hits at most
one mesh emitter.

**The trap:** every MIS weight must use that combined pdf. `misBalanceWeight(bsdfPdf,
envPdf)` at `util-raytrace.glsl:1454` and `:1517` must become
`misBalanceWeight(bsdfPdf, pickPdfEnv * envPdf)`. Missing the `pickPdf` factor is a
silent systematic darkening no NaN probe catches — tier 2.4/2.5 exist for it.

### 3.6 Extensibility (not implemented now)

A directional/quad/point light becomes a sentinel entry (`triangleCount == 0`) with its
own sample/pdf branch, as cull's sun and sphere lights already were. Delta lights return
`pdf = 0` from the backward lookup so the BSDF technique always gets weight 1 on them.

At scale — many emissive meshes, e.g. bistro — a flat power CDF ignores distance and
orientation and degrades badly. The standard answer is a light BVH: Conty Estevez &
Kulla 2018, which is what Cycles' light tree implements (two-level: a top-level tree over
lamps and mesh lights, and a per-mesh subtree whose leaves are triangles). Out of scope,
but the `emitterSample`/`emitterPdf` interface is the seam it would land behind.

---

## 3.7 The shadow-ray origin trap (found and fixed, 2026-08-05)

The nee shadow ray offsets its origin along the geometric normal to avoid
self-intersection, then needs an end margin so the light does not occlude
itself. The obvious form of that margin is wrong:

```glsl
// WRONG: es.direction/es.distance are measured from the shading point, so the
// ray leaves the offset origin parallel-shifted and never aims at the sample
const f32 shadowMaxDist = es.distance - 0.001f;
utilTraceRay(shadowOrigin, es.direction, shadowMaxDist, ...);
```

Nudging the origin by `delta` along the normal moves the crossing of the
emitter's plane by `delta / dot(dir, n_emitter)` — unbounded as the connection
approaches grazing — while the margin subtracts a constant. The ray therefore
crosses the emitter's plane *before* its cutoff and reports occlusion. For a
`0.0005` nudge against a `0.001` margin the failure sets in below
`dot(dir, n) = 0.5`, i.e. **every connection under 30 degrees of elevation was
silently discarded as shadowed**.

Measured on the tier-3 floor/emitter scene: 13.4% of all traced shadow rays
rejected, every one of them blocked by the emitter's own geometry and none by
anything else, costing 1.3% of the image's energy.

The fix is to derive both the direction and the distance from the offset
origin, so the ray genuinely ends at the sampled point and a relative epsilon
suffices:

```glsl
const f32v3 shadowToLight = es.position - shadowOrigin;
const f32 shadowMaxDist = length(shadowToLight) * (1.0f - 1e-4f);
const f32v3 shadowDir = shadowToLight / length(shadowToLight);
```

The estimator keeps `es.direction` — `f`, the pdf and the mis weight all belong
to the shading point, not to the offset origin. After the fix the occlusion
rate is 0.00012% and `contributed` equals `passed f/cos` exactly, as an
occluder-free scene requires.

**Verified across the whole domain.** After the fix, the closed-form error is flat at
−0.64% ± 0.005% over six configurations spanning four orders of magnitude of subtended
solid angle and both sampling branches, and the nee-on/nee-off gate agrees to 0.93 sigma
against its own measured noise. The residual −0.64% is a *pre-existing* deficit present
with nee entirely off (see below), not attributable to this feature.

**A second bug lived in the test's own reference**, and it masqueraded as a renderer
defect for two rounds. `polygon_irradiance_at` summed `theta_i * dot(n, e_i)` in f32.
Those terms largely cancel — their sum is the projected solid angle, orders of magnitude
below any individual term — and for a small emitter the corner vectors are nearly
parallel, so the f32 cross products lose most of their significant digits exactly where
the reference matters most. It read as a −5.4% error in the area sampling branch.
Recomputed in double, with `2*asin(chord/2)` instead of `acos(dot)`, the same
configuration reads −0.635%: indistinguishable from every other point on the sweep.
A closed-form reference is only a reference where its own conditioning holds.

**Note the same shape exists in the env nee block**, which offsets identically
and traces to `999999.0f`; an infinite-distance ray has no end margin to get
wrong, so it is not affected. Any future finite-distance connection (analytic
lights, portals) has to derive its ray from the offset origin.

## 4. Correctness traps to design against

1. **Delta bounces must skip MIS** — §3.4(b). Confirmed by both references and by cull's
   own comment. Manifests as thin glass looking too dark.
2. **Mixture/pick pdf desync** — §3.5. The `pickPdf` factor must appear in *every* MIS
   weight, forward and backward.
3. **`previousVertexSampledLights`** — §3.4(b). Energy loss on paths the light technique
   could not generate.
4. **Flat vs meshlet-local primitive numbering** — §3.1. Forward sampler and backward
   lookup must agree.
5. **Sidedness must match on both halves** — §3.3. A live bug class in Blender today.
6. **mor reserves material index 0** as an unreferenced default (per the resolved
   material-color-load finding). Reach materials through `flatMeshlets[prim] →
   meshlet.materialIndex`.
7. **Instanced emitters** share a BLAS but not a world area. Keying on `modelDrawIndex`
   handles it; a grass-field-scale scene with an emissive tile can blow past
   `kMaxEmitterLights` — log the clip.
8. **Quad seams** — a rectangle is two triangles; a sampler or pdf discontinuous across
   the shared diagonal shows as a faint gradient.
9. **Degenerate triangles** — zero area poisons the CDF; drop at build, and return pdf 0
   for `Ω == 0` in the spherical path.
10. **Self-intersection** at both ends of the shadow ray.
11. **Near/large emitters** — the area-sampling variance blowup §3.3 exists to fix.
12. **Zero emitters must cost nothing** — no rng draws when the list is empty, so a
    non-emissive scene renders bit-identically with the feature on and off (t3.6).

---

## 5. Testing plan

Conventions follow the existing suite: doctest cases in `unit-test/src/`, compute shaders
in `unit-test/shaders/`, heatmaps and validation renders into `unit-test/output/`,
chi-square on the `util-sampling-bins.glsl` 40×80 sphere grid with the Pearson +
Wilson–Hilferty convention from `test-vdb-phase.cpp:142-234`.

New files: `test-emitter-tables.cpp` (tier 0), `test-emitter-sampling.cpp` (tier 1–2),
`test-emitter-nee-scenes.cpp` (tier 3).

### Tier 0 — CPU table build (no GPU)

| id | test | catches |
|---|---|---|
| 0.1 | world area vs analytic under identity / uniform-scale / non-uniform-scale / rotate / shear | transform applied in the wrong space |
| 0.2 | CDF monotonic, last entry exactly 1.0, `power` sums to `totalPower` | builder invariants |
| 0.3 | 10⁷ CPU draws through the CDF, chi-square against the weights | pick distribution |
| 0.4 | synthetic scene with known emissive materials on a subset of meshlets → exact expected record set, and `primitiveIndexBase`/`triangleCount` ranges match the *flat* numbering | trap 4, trap 6 |
| 0.5 | degenerate input: zero-area meshlets, zero emission with an emissive texture, all-black scene (empty list), single triangle, list overflowing `kMaxEmitterLights` | trap 7, trap 9, empty-list path |
| 0.6 | same mesh instanced twice at different scales → power ∝ *world* area | trap 7 |
| 0.7 | material emission edited → next frame's rebuild reflects it | live-edit contract |

### Tier 1 — GPU sampler vs pdf

| id | test | catches |
|---|---|---|
| 1.1 | **round-trip**: recompute `emitterPdf` from each sample's `(model, prim, dist, dir, normal)`, require it matches the returned pdf to ~1e-6 relative. Swept over all triangles, both faces, grazing angles | the property MIS unbiasedness rests on |
| 1.2 | chi-square sampler-vs-pdf on the sphere grid. Configs: single triangle; two at different distances; a quad as two triangles (**seam**, trap 8); a closed emissive box around the point; a near-edge-on triangle | density and area→solid-angle errors |
| 1.3 | `emitterPdf` == 0 for a non-emissive triangle and for an emissive one absent from the list | stale/capped-list unbiasedness |
| 1.4 | dense NaN/degenerate sweep in the style of `test-environment-sampling.cpp`: point in the emitter's plane, at a vertex, at the centroid, at distance 1e-6 and 1e6; ξ at (0,0)/(1,1)/(1,0)/(0,1)/(0.5,0.5); denormal areas | traps 9-11 |
| 1.5 | sample from both sides of one triangle; behavior matches the §3.3 sidedness decision on *both* the sampler and the pdf | trap 5 |
| 1.6 | a single meshlet mixing one huge and many tiny triangles: still unbiased, and the variance penalty is measured and recorded | the §3.2 uniform-within-meshlet trade-off, quantified rather than assumed |
| 1.7 | **both hybrid branches normalize correctly**: `E[1/pdf]` against the analytically computed subtended solid angle (Van Oosterom–Strackee on the CPU), with an assertion that the intended branch was actually taken | a density off by a constant factor, in either branch |

**A sphere-grid chi-square cannot referee the area branch, and this is structural**, not a
tuning problem: a `40x80` bin is ~6e-3 sr, and an emitter only takes the area branch when it
subtends *less than* 3e-4 sr — roughly 20x smaller than one bin. The same limit bites whenever
the sampler concentrates draws onto a sub-bin feature: with 75% of 10^6 draws landing on three
triangles spanning ~2 bins, the test is implicitly demanding the expected-side subgrid resolve
their silhouettes to ~0.2%, and raising the subgrid 9x moved the statistic by 6% — the
signature of an edge-position error no affordable subgrid fixes. `E[1/pdf]` vs the analytic
solid angle is the resolution-independent replacement and is what tests 1.6 and 1.7 use.
Keep chi-square for emitters that comfortably span several bins.

### Tier 2 — estimator vs analytic ground truth

| id | test | catches |
|---|---|---|
| 2.1 | Lambertian receiver under an axis-aligned emissive rectangle vs the closed-form Lambert rectangular-source irradiance, run **three ways**: NEE-only, BSDF-only, MIS. All must agree within MC error; MIS variance ≤ both | a dropped 1/p, a missing cosine, a wrong MIS weight — at once |
| 2.2 | emissive sphere enclosing the point, uniform emission → irradiance = π·L, **invariant under tessellation** (12/24/96 rings) | area/pdf conversion, seams |
| 2.3 | **MIS partition sweep**: roughness 0.01→1.0 × light solid angle small→large. NEE-only, BSDF-only, MIS means must coincide across the grid | a mis-weighted branch biases only one corner |
| 2.4 | env map **and** emissive mesh present: `combined == env-only + mesh-only` at matched spp | trap 2 |
| 2.5 | vary the emitter/env power split over {0.1, 0.5, 0.9}: mean invariant | `pickPdf` not divided out, or not folded into the MIS pdf |

### Tier 3 — path-level and scene renders

Every case is the same assertion — **NEE-on and NEE-off converge to the same image** —
applied to a different corner. Needs the `meshNeeEnabled` toggle (§6) to exist first.

**These must not run through the viewer's accumulation path.** `pt-accumulate.comp:127`
averages `sum(rgb * a) / sum(a)`, where `a` is the sample-validity flag — only
light-carrying samples enter the denominator (the valid-sample spp convention). So NEE-off
converges to `E[L | L > 0]` while NEE-on, where nearly every path picks up some
contribution, converges to `E[L]`. The two are normalized over different sample
populations, so a mean comparison between them measures the convention, not the
estimator. Measured on `EmissiveStrengthTest.glb` at 256 spp this reads as NEE-on being
55% "darker", which is not a bug and not evidence of one either way.

Tier 3 therefore belongs in the unit-test harness against a shader that averages plainly
(`radianceSum / sampleCount`, as `cornell_openpbr_render.comp` already does).

**Cross-check (2026-08-04).** Neither reference renderer normalizes radiance this way, and
the difference is structural rather than stylistic:

- **PBRT-v4** accumulates `pixel.rgbSum += weight * rgb; pixel.weightSum += weight` for
  *every* camera sample and finishes with `rgb /= weightSum`. A zero-radiance sample still
  contributes its filter weight to the denominator.
- **Cycles** divides by `pass_sample_count` — samples *taken* for that pixel, per-pixel so
  adaptive sampling can stop pixels at different counts — and keeps coverage in a separate
  channel entirely (`film_write_transparent`, with the comment "we accumulate
  transparency = 1 - alpha in the render buffer").

So both keep two distinct quantities that ponder's `irradiance.a` currently collapses into
one: *how many samples did this pixel take* (the radiance denominator) and *how covered is
this pixel* (compositing alpha). Averaging only nonzero samples estimates
`E[L | L>0] = E[L] / P(L>0)`. The unbiased way to avoid spending work on low-contribution
paths is Russian roulette with its `1/(1-q)` compensation — which this path tracer already
does for throughput.

Splitting the two is out of scope for NEE, but worth deciding on: it is the difference
between the viewer's image being an estimate of `E[L]` and an estimate of `E[L]/P(L>0)`.

| id | test | catches |
|---|---|---|
| 3.1 | cornell box with the existing emissive ceiling quad, on vs off at high spp: per-pixel mean difference under tolerance + a difference heatmap | the headline correctness test |
| 3.2 | equal-spp RMSE vs a high-spp reference: NEE-on ≥3× lower on a small-light cornell | "correct but useless"; also the case §3.3's sampler choice decides |
| 3.3 | cornell light shrunk to 1% area: off is pure noise, on converges | the point of the feature |
| 3.4 | emissive quad behind a smooth dielectric slab, on/off means match | §3.4(b) occlusion/MIS consistency |
| 3.5 | emitter seen from a **subsurface exit vertex** and from inside a **vdb medium**, on/off means match | sites (c)/(d) — no prior art, highest risk |
| 3.6 | zero-emitter scene: **bit-identical** on vs off, zero shadow rays. Plus a scene whose only light is a mesh emitter reached solely via NEE — assert the `hit`/spp alpha is set (valid-sample convention) | trap 12, spp accounting |
| 3.7 | emissive quad shaded from both sides, matching the §3.3 decision and the analytic value on whichever sides emit | trap 5 |
| 3.8 | N instances of one emissive mesh at different transforms: total == sum of the N rendered individually | trap 7 |
| 3.9 | strongly spatially-varying emissive texture: on/off means match | the factor-only `power` approximation must not bias radiance (§3.2) |
| 3.10 | edit emission without rebuilding, and overflow `kMaxEmitterLights`: means still match a from-scratch reference | the stale/capped-list contract |
| 3.11 | real glTF models through `furnace_model_render.comp` with emission forced on, under the existing NaN probe; include grazing/degenerate geometry | NaN/inf, fireflies |
| 3.12 | emitter partially occluded by an alpha-BLEND surface: on/off means match | stochastic shadow-ray coverage × NEE |
| 3.13 | **delta-lobe sweep**: emitter seen through thin-walled transmission (`geometryThinWalled > 0`), through a roughness-0 specular reflection, and via the TIR fallback direction. on/off means must match | trap 1 — cull's documented "thin glass renders too dark" bug |

### Tier 4 — regression & performance

- 4.1 golden validation renders (cornell + one stage scene) into `unit-test/output/`,
  README updated to the existing image-per-feature convention.
- 4.2 no measurable cost in the zero-emitter case; log emitter count, clip events, and
  per-frame table build time so a pathological scene is visible rather than mysterious.

---

## 6. Implementation phases

| phase | content | tests green at the end |
|---|---|---|
| ~~P0~~ **done** | `emitter-tables.{hpp,cpp}` (power CDF, per-instance ranges); `emitter-shared.h`. `emitterLightOffset/Count` deferred to P2, where they first get read | tier 0 — 10 cases |
| ~~P1 + P1.5~~ **done** | `util-emitter.glsl` — pick/sample/pdf/backward-lookup, hybrid area + spherical-triangle sampling landed together | tier 1 — 9 cases |
| ~~P2~~ **done** | surface-vertex nee block, mis on the emission branch with delta exclusion, path state, per-frame cpu list build, `--mesh-mis` + imgui toggle | tier 2 — 2 cases |
| ~~tier 3~~ **done** | production-tracer render harness on hand-built mor buffers, closed-form reference, nee connection instrumentation | tier 3 — 7 cases |
| ~~P3~~ **done** | env map folded into the light list as an `EMITTER_KIND_ENVIRONMENT` entry; one pick, one shadow ray; `previousEnvPickPdf` path state through every env-side mis weight | tier 2.4-2.5 — 3 cases (§9) |
| P4 | subsurface exit + vdb medium vertices | tier 3.4-3.5 |
| P5 | golden images, logging, README TODO line removed | tier 3.6-3.12, tier 4 |

Prerequisite work (§2) lands before P2.

---

## 7. Decisions needed before P0

1. **Per-meshlet emitter records with uniform triangle pick** (recommended — cull's proven
   shape, small enough to rebuild per frame, reuses `scene_meshlet_area_info`) vs
   per-triangle records with an alias table (lower variance on uneven meshlets, larger
   table, needs a search for the backward lookup).
2. **Two-sided emission** (recommended — matches ponder's current BSDF-hit branch and
   glTF's double-sided default) vs cull's one-sided vs Cycles' per-material
   front/back/both/none.
3. **Area sampling in P1, hybrid spherical in P1.5** (recommended) vs going straight to
   the hybrid vs staying on area sampling permanently.
4. **Env map folded into the power-weighted light list** (recommended — what cull and
   Cycles both do) vs a fixed technique-selection probability.
5. **Radiance at the sampled point evaluated from the texture** (recommended — diverges
   from cull, which uses the precomputed per-emitter average, and is what keeps the two
   MIS techniques exactly consistent for textured emitters) vs cull's cheaper average.

---

## 8. Cross-check against other renderers

### 8.1 cull — ponder's own prior implementation (the most relevant source)

`~/repo/cull` has a complete, working emissive-mesh NEE that was deleted from ponder on
2026-07-17 during the PT reset. It is a closer match to this codebase than any external
reference: same `GpuResolveModelIndirect`, same flat primitive numbering, same
`utilTraceRay`, same OpenPBR eval/pdf calls.

What it establishes:

- Per-emissive-**meshlet** records with `primitiveIndexBase`/`triangleCount` ranges,
  rebuilt every frame, power-weighted CDF, binary-search pick. → revised §3.2.
- Non-mesh emitters (sun, sphere lights) live in the *same* list as
  `triangleCount == 0` sentinels, so one power-weighted pick covers everything. → revised
  §3.5 and §3.6.
- Backward lookup by per-model emitter range + linear scan over
  `[primitiveIndexBase, +triangleCount)`. → §3.2.
- The delta-bounce exclusion, with the concrete failure it fixes: thin-walled
  transmission samples `wo = -wi` exactly, "blending with a nonzero lightPdf
  systematically discounts direct light seen straight through thin glass". → trap 1,
  test 3.13.
- The flat-vs-meshlet-local numbering warning. → trap 4.
- One-sided emission (`cosAtLight > 0` on both the forward sample and the backward
  lookup) — the one place I recommend diverging, since ponder's current BSDF-hit branch
  is two-sided.

What it does **not** cover, and so gets no prior art: NEE from a subsurface exit vertex,
NEE from a vdb medium vertex, and env-map NEE as a list entry (cull had a Preetham sky +
sun, not an importance-sampled HDR map). Those three are the highest-risk parts of P3/P4.

One structural difference that cannot be ported directly: cull traced the BSDF ray
*inside* the same function that shaded the vertex, so the emitter-hit MIS weight was
computed locally with `bsdfPdf` in scope. ponder split that into `utilIrradianceWalk`, so
the same weight now needs `hitModelDrawIndex`, `hitPrimitiveIndex`, hit distance and two
flags threaded as path state. §3.4(b).

### 8.2 Blender Cycles

- **Hybrid triangle sampling**: solid-angle (spherical triangle) sampling when
  `longestEdge² > distanceToPlane²`, uniform area otherwise. Solid angle by Van
  Oosterom–Strackee: `Ω = 2·atan2(mixedProduct, 1 + cos a + cos b + cos c)`, `pdf = 1/Ω`,
  with an explicit `Ω == 0 → pdf = 0` guard. → §3.3, trap 9.
- **Sidedness is per-material and four-valued** — front / back / both / none, enforced via
  `SD_MIS_FRONT` / `SD_MIS_BACK`, with the sampled side rejected when its flag is unset.
  The side is chosen by the sign of `distance_to_plane`. → §3.3, decision 2.
- **Delta-bounce exclusion via path state**: `mis_ray_pdf` carries the previous bounce's
  BSDF pdf, `min_ray_pdf` tracks the running minimum, and MIS is skipped when it is zero.
  `light_sample_mis_weight_forward_surface` reads both. Independent confirmation of
  trap 1 and of the path-state threading in §3.4(b).
- **One combined light distribution** including the background. → §3.5.
- **Light tree** for many lights: two-level, top-level over lamps and mesh lights, a
  per-mesh subtree whose leaves are triangles; traversal picks an emitter with one random
  number. → §3.6 future work.
- **Cautionary data point**: Blender #115997 — emission-sampling settings render
  differently with the light tree on vs off. Sidedness × sampler interaction is a real
  bug class in a mature renderer, which is why test 1.5 checks the sampler and the pdf
  agree about sidedness rather than only checking rendered output.

### 8.3 PBRT-v4

- Spherical triangle sampling when the subtended solid angle is in **`[3e-4, 6.22]` sr**,
  uniform area sampling outside that window (numerical stability at both ends) —
  a concrete, citable threshold to adopt in P1.5.
- With a reference normal, PBRT additionally **warps by the cosine**: bilinear fit over
  `AbsDot(ns, wi)` at the four spherical-triangle corners, clamped to a 0.01 floor,
  sampled with `SampleBilinear` and the pdf multiplied by `BilinearPDF`. A cheap extra
  variance win, and cleanly separable — worth trying only after 1.7 is green, since it
  makes the pdf non-constant again.
- Area→solid-angle conversion is the same `d² / |cos θ_l|` factor cull uses.

### 8.4 Net effect on the plan

Revised: emitter granularity (§3.2), env folded into the light list (§3.5), sampling
scheme staged area → hybrid spherical (§3.3), backward lookup by range scan (§3.2).
Added: delta-bounce exclusion as a first-class requirement (trap 1, test 3.13), the
flat-numbering trap (trap 4), the sidedness-agreement trap (trap 5), tests 1.6/1.7/3.13.
Unchanged and independently confirmed: the `(modelDrawIndex, primitiveIndex)` emitter
identity, the pick-pdf-in-every-MIS-weight trap, the `previousVertexSampledLights`
requirement, and the on/off mean-equality test strategy.

**Sources**

- [Cycles `kernel/light/triangle.h`](https://projects.blender.org/blender/blender/raw/branch/main/intern/cycles/kernel/light/triangle.h)
- [Cycles `kernel/integrator/shade_surface.h`](https://projects.blender.org/blender/blender/raw/branch/main/intern/cycles/kernel/integrator/shade_surface.h)
- [Blender #115997 — emission sampling differs with light tree on vs off](https://projects.blender.org/blender/blender/issues/115997)
- [Cycles: Many Lights Sampling (#77889)](https://developer.blender.org/T77889)
- [Cycles light tree instancing support (#106683)](https://projects.blender.org/blender/blender/pulls/106683)
- [PBRT-v4 — Triangle Meshes / sampling](https://pbr-book.org/4ed/Shapes/Triangle_Meshes)
- [Conty Estevez & Kulla, *Importance Sampling of Many Lights with Adaptive Tree Splitting* (HPG 2018)](https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf)

---

## 9. P3 as landed (2026-08-05)

**Shape.** `GpuEmitterLight` grew a `kind` field (`EMITTER_KIND_MESH` /
`EMITTER_KIND_ENVIRONMENT`), and `emitter_lights_append_environment` puts the
environment map in the same power-weighted list as the emissive meshlets,
appended after every mesh entry so no model's `emitterLightOffset/Count` range
can cover it and the backward hit lookup can never match it. Its power is
`EnvironmentTables::luminanceIntegral * envIntensity` — `\int luma(L) d\omega`
over the sphere, a new field on `EnvironmentTables`. That is an
irradiance-scale number weighed against a mesh emitter's `area * radiance`, so
the two are not strictly commensurable; it steers the pick and nothing else,
which is what test 2.5 pins.

**One pick, one shadow ray.** The surface vertex now draws four uniforms once,
picks one entry, and takes either the env branch or the mesh branch. Where the
env is *not* in the list (a harness uploading a mesh-only list, or mesh nee
off with no env), env nee still stands on its own with the whole pick mass —
so the rng stream, and therefore every tier 0-3 result, is bit-identical to
P2 for a mesh-only scene, and identical to the pre-P3 behaviour for an
env-only scene.

**`previousEnvPickPdf` replaced `envNeeActive` in every backward weight.**
`misBalanceWeight(bsdfPdf, envPdf)` was wrong in three separate ways at once,
and the pick factor was only the first:

1. it missed `pickPdf` (the trap §3.5 was written for);
2. it fired after a **delta bounce**, which nee cannot reproduce — the same
   defect trap 1 documents on the mesh side, present on the env side the whole
   time;
3. it fired after a vertex that ran **no env nee at all**.

All three collapse into one number: the environment's share of the pick at the
vertex the segment left from, 0 where the technique did not run. `misBalanceWeight(a, 0) == 1`
does the rest, so the guard *is* the value.

**Two path-state fixes fell out of threading it.** Both are pre-existing and
neither had a test:

- the **subsurface walk's exit vertex** is where the next segment starts, but
  it inherited the *entry* vertex's `vertexSampledLights`. An emissive
  triangle hit straight after the exit was balanced against a mesh-light
  density that never sampled it. Now the exit reports its own state: no mesh
  nee (that site is P4), env nee with pick pdf 1.
- a **vdb medium scatter** is likewise the vertex the next segment leaves
  from, and the loop carried the last *surface* vertex's flags across it.

**Tests** (`test-emitter-nee-scenes.cpp`):

- *env and mesh emitters combine linearly* — transport is linear in the
  sources, so `combined == mesh-only + env-only` with the emissive quad acting
  as the identical (black) occluder in all three. Measured 1.1 sigma.
- *the env/mesh power split does not move the mean* — the env entry's power
  forced to a 0.1 / 0.5 / 0.9 share of the pick. A pick probability that does
  not divide back out biases in proportion to the share and cannot survive a
  9x swing in both directions. Measured max 0.6 sigma.
- *the split between two mesh emitters is inert* — the same sweep with the
  environment out of the picture entirely, so a trend in the case above can be
  attributed. This one exists because the first version of the split sweep
  *did* look like a monotone trend, and this is what showed the trend was in
  neither the pick nor the env sampler.

**A methodology note the split sweeps forced.** The half-spread of a *pair* of
realizations is a one-degree-of-freedom estimate of sigma: it lands anywhere
from ~0 to ~3x the truth. That is fine for gating a percent-scale
discrepancy, and useless for gating a sub-0.1% one — the first split sweep
failed at 5 sigma on a 0.055% spread purely because one configuration's two
realizations happened to land close together. Four realizations and a proper
standard error settle it, and quadrupling the sample count halved the spread,
which is the `1/\sqrt{N}` signature of noise rather than bias. Any future
sub-0.1% gate in this file needs `render_mean`, not a pair.

**One micro-inconsistency found and left alone.** The forward nee pdf is
evaluated at the exact surface point (`itOri`, before the end-of-bounce
normal nudge), while the backward pdf is evaluated at `walkSegmentOrigin` —
the same point *after* the `0.0005` nudge. The two mis weights for one
direction therefore do not sum to exactly 1; the residual is
`O(nudge / distance)`, about `4e-5` relative at the tier-3 scene's scale,
an order of magnitude under the noise floor of any test here. Recorded
because it is the kind of thing that grows teeth in a scene with emitters
at millimetre distances.

**Not covered by P3.** Mesh nee at the subsurface-exit and vdb-medium
vertices is still P4; those two sites sample the environment alone with pick
pdf 1, which is self-consistent (both halves of their mis agree) but leaves
mesh emitters there to the bsdf technique only. In stage mode the light list
is rebuilt on structural edits rather than per frame, so changing
`envIntensity` or the mesh-nee toggle afterwards leaves the pick share stale —
a variance cost only, since forward sampling and the backward lookup both read
the same uploaded cdf.
