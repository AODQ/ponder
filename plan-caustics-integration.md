# Caustics integration — what stands between the solver and a lit pixel

Successor to `plan-slang-caustics.md`, which took the project through its Stage 2: a
differentiable Snell's-law constraint, a Newton solver generic over a surface interface,
a gradient check against finite differences, and a diagnostic pass that counts how often
the solver lands on a bend point. None of that carries radiance.

This plan is an audit, not a design. It names every gap between what exists today and
caustics in the render, with the file and line each one lives at, and it flags the two
decisions that have to be made before any of the rest can be ordered sensibly.

Scope boundary inherited from the predecessor: this document stops at Stage 3 (caustics
appear on screen) and names Stage 4 (unbiased, MIS) and Stage 5 (reflective) only where
an earlier choice would foreclose them.

---

## 0. What exists today

| piece | file | state |
|---|---|---|
| constraint + Newton solver, generic over `ISurface` | `lib/ponder/shaders/util-manifold.slang` | works; triangle and sphere surfaces |
| mesh surface with ray-cast reprojection | `lib/ponder/shaders/util-manifold-mesh.slang` | works on a single-instance mesh |
| gradient check, solve-back tests | `unit-test/shaders/slang/manifold.slang`, `unit-test/src/test-slang.cpp:35,75` | covers Stages 1-2 |
| convergence probe, separate dispatch | `app/viewer/shaders/manifold-probe.slang` | diagnostic only |
| flat-colour painter | `app/viewer/shaders/manifold-shade.slang` | debug only, carries no radiance |
| probe/shade render nodes, ImGui readout | `app/viewer/main.cpp:1789-1901`, `:3050` | scaffolding |

Stages 0-2 of `plan-slang-caustics.md` are done. Everything in §§1-9 below is Stage 3 and
the scaffolding Stage 3 needs.

---

## 1. Gap A — the solver handles one bend point; glass needs two

`manifold.Path` (`util-manifold.slang:54`) is `{pointWi, pointWo, eta}`: a single specular
vertex. That is Zeltner's `manifold_ss` case, and the predecessor plan is scoped to it
throughout — its §9 speaks of "the constraints across all three vertices", meaning
receiver, bend, light.

The reference scene already in the repo is not that shape.
`unit-test/src/test-dielectric-scenes.cpp:1285` (glass ball over a diffuse floor) carries a
**two**-interface path: floor, entry refraction, exit refraction, light. The solver as
written cannot express it.

**Decided 2026-10-25: chain of two.** `Path` becomes a vertex array, the Newton system
becomes block-tridiagonal across both bend points, and the generalized geometric term of
§6 grows correspondingly. This is what the glass ball and the glass dragon need, and it
keeps `test-dielectric-scenes.cpp:1285` usable as the §9 reference instead of requiring a
new single-interface scene. §1a specifies what the change entails.

§§2-7 apply either way; the chain only makes §6 harder.

---

## 1a. The chain-of-two solver, concretely

Six changes, in dependency order. All of them in `lib/ponder/shaders/util-manifold*.slang`.

**Fix the chain length at two, as a compile-time constant.** Not a runtime-length array.
Slang's autodiff over a dynamic loop needs `[MaxIters]` and gets fragile fast; two bend
points unrolled is four scalar unknowns and a 4x4 solve, both small enough to write out.
Name the constant so a later third interface is a localized edit rather than a rewrite.

**`Path` carries per-interface eta.** For solid glass the entry is air-to-glass and the
exit is glass-to-air, so the two interfaces take reciprocal relative iors — a single
`eta` field cannot express the path. This is the easiest thing on the list to get subtly
wrong, because an inverted exit eta still converges, just to the wrong surface point.

**Watch the exit vertex's normal orientation.** `constraint_half_vector`
(`util-manifold.slang:138`) projects `h` onto the tangent frame, and that projection is
sign-agnostic — but `h = wi + eta * wo` is not. At the exit the ray arrives from inside
the medium, so whichever convention you pick (normal-facing, or eta-reciprocal) has to be
applied consistently at both vertices. The cross-check against a forward-refracted path is
what catches an inconsistency here; extend `path_refracting_forward`
(`unit-test/shaders/slang/manifold.slang:71`) to refract twice.

**The constraint becomes `f32v4 -> f32v4`,** two components per bend point. Each bend
point's constraint couples only to its neighbours, so the Jacobian is block-tridiagonal:
for N=2, two 2x2 diagonal blocks and two 2x2 off-diagonal blocks. At this size solve the
4x4 directly — a general block-tridiagonal sweep is not worth writing for two blocks.
Forward-mode autodiff costs one pass per input, so the full Jacobian is four `fwd_diff`
calls rather than the current two. Still cheap.

**`MeshPointState.rayOrigin` has to stop being a constant.** It is set once at
construction (`util-manifold-mesh.slang:107`) and read as the reprojection ray's origin
(`:146`). In a chain, bend point 2's reprojection ray originates at bend point 1's
*current* position, which moves every Newton iteration. So `advance` takes the previous
vertex's position as a parameter instead of reading a stale field, and the chain's
`advance` steps the vertices in order so each sees its predecessor's new position.
Getting this wrong gives a solver that converges on flat casters and fails mysteriously
on curved ones.

**Seed by tracing, not by drawing twice.** Two independently drawn random triangles will
essentially never form a plausible chain. Trace from the receiver toward a random point
on the caster to place vertex 1, then refract through it and trace again to place
vertex 2 — which is also where the caster-only hit test of §2 naturally lives.

---

## 2. Gap B — caster addressing is a placeholder

`manifold.MeshData` (`util-manifold-mesh.slang:23`) is a private duplicate of
`GpuResolveModelIndirect`, and `app/viewer/main.cpp:1818` fills it from `sScene.bufs` with
`modelMatrix` set to identity and the scene-wide flat index buffer — that is, *the whole
scene is one caster*. Three concrete defects follow.

1. **The instance is dropped.** `SurfaceMesh.advance` (`util-manifold-mesh.slang:139`)
   reads `rq.CommittedPrimitiveIndex()` and ignores `CommittedInstanceID()`. The primitive
   index is BLAS-local, so with more than one instance in the TLAS the re-fetch reads a
   different mesh's triangle under the wrong transform. Single-model scenes hide this
   completely.
2. **Any hit is accepted.** A reprojection ray that grazes past the glass and lands on the
   floor continues the walk on the floor, and the solve can still report `converged`. The
   convergence rate the probe prints is therefore not measuring what it appears to.
3. **The fetch is duplicated.** `utilMeshAttributeDataFromIndices`
   (`lib/ponder/shaders/util-mesh-raytrace.slang:26`) already resolves
   `(modelDrawIndex, primitiveIndex, barycentric)` into world-space position, shading
   normal and the instance transform, including the inverse-transpose normal matrix that
   `mat3_inverse_transpose` (`util-manifold-mesh.slang:52`) re-derives by hand.

The fix is one shape: carry `modelDrawIndex` in `MeshPointState`, drop `MeshData` and the
private fetch in favour of the shared helper, and reject a reprojection hit whose instance
id is not the caster's.

---

## 3. Gap C — no caustic-caster flag

Nothing in `lib/mor` or `lib/stage` marks an object as a caustic caster, which
`plan-slang-caustics.md` §8.1 asks for and both Mitsuba and Cycles carry. The probe draws
a uniformly random triangle out of the entire scene
(`app/viewer/shaders/manifold-probe.slang:80`), so in any scene that is not a lone glass
object the overwhelming majority of seeds are hopeless by construction.

Needed: a bit on the material or the stage instance, plumbed through to a per-caster
triangle list — or, as a first cut, a list of caster draw indices the seeder picks from
before picking a triangle within one.

Until this and §2 land, the convergence percentage in the ImGui readout
(`app/viewer/main.cpp:3050`) should not be read as a quality signal.

---

## 4. Gap D — the receiver is a g-buffer texel, not a path vertex

The probe runs as a full-screen dispatch scheduled after the resolve node
(`app/viewer/main.cpp:1789`), reading `gbufferPos` and `gbufferNor`. That buys: primary
hits only, one seed per pixel per frame, no path throughput, no material at the receiver,
and nothing at all at secondary or subsurface-exit vertices.

`plan-slang-caustics.md` §5 argued for a separate pass on two grounds. The first — "do not
port the path tracer to Slang" — expired with the 2026-09-29 shader port; the path tracer
is Slang now and can `import "util-manifold-mesh"` directly. The second still holds:
divergence. A Newton loop with ray casts inside it, inlined into the megakernel, is a
small number of threads doing twenty iterations while the rest of the warp waits.

**Decided 2026-10-25: inline in the NEE block** of `utilIrradiance`
(`app/viewer/shaders/util-raytrace.slang:693`). `itOri`, `itFrame`, `itCoatFrame`,
`material`, `irradianceThroughput`, `state` and `unf.emitterLights` are all already in
scope there. By a wide margin the smallest diff, and the only version that reaches
non-primary vertices without new buffers. The rejected alternative — a path-vertex seed
buffer with append/compaction, throughput carried per record and a deferred write-back —
stays available if the measurement below says it is needed.

Three consequences to plan for:

- **Gate it hard.** `unf.manifoldNeeEnabled` (§8) is the branch, and the caster flag (§3)
  is the second gate. Without both, every diffuse vertex in the scene attempts a Newton
  solve and the cost lands on scenes that have no glass in them at all.
- **Measure divergence deliberately, not by feel.** The tile budget already reports
  per-tile gpu-ms (`sMsPerTileEma`, `main.cpp:1439`). Record it on the glass-ball
  scene with the feature off, then on, before tuning anything. That number is also what
  decides whether the two-pass design comes back.
- **Bound the work per vertex.** `Settings.maxIter` defaults to 20 and `maxBacktrack` to
  8 (`util-manifold.slang:66`), which is a reasonable offline budget and a poor
  interactive one. Expect to want a smaller interactive cap, and to find the right value
  from the measurement above rather than guessing it now.

---

## 5. Gap E — no light sampling

`pointWo` is an ImGui-aimed constant (`app/viewer/main.cpp:941`, `sManifoldPointWo`,
exposed at `:2794`). It has to come from the emitter list instead: `utilLightPick`
(`util-raytrace.slang:335`) and `utilEmitterSampleFromPick`, the pair the mesh NEE block
at `:693` already uses.

Note that `GpuManifoldProbePC` (`app/viewer/shaders/manifold-probe-pc.h:33`) carries no
`GpuGlobalPc`, so the probe cannot currently see `unf.emitterLights`, `emitterLightCount`
or `envLightIndex` at all. That is another point in favour of §4's inline option, which
inherits all of them.

---

## 6. Gap F — no contribution

`manifold-shade.slang:41` adds a flat debug colour to the pixel. Turning a found bend point
into radiance (predecessor §9) needs all of the following, none of which exists:

- **Receiver BSDF.** `openPbrEvaluateF` and `openPbrEvaluatePdf` toward the direction of
  the bend point, exactly as the NEE block does toward `es.direction`.
- **Specular transmission at each bend point.** Requires loading the caster's material at
  the solved barycentric — a second use for the shared fetch of §2.
- **Emitter radiance.** `utilEmitterSampleRadiance` (`util-raytrace.slang:247`), which
  reads the emitter point's own material rather than the light list's steering value.
- **Two visibility segments**, receiver to bend and bend to light. Neither is traced
  today; `manifold-probe.slang:103-104` checks hemisphere signs only.
- **The generalized geometric term `G`.** Nothing exists. Two sub-problems:
  `solve` (`util-manifold.slang:161`) discards the converged Jacobian and should return it
  in `IterationResult`; and `G` additionally needs the constraint's derivative with
  respect to the *light* point, which means a second `fwd_diff` seeding against
  `path.pointWo`. `Path` is presently a plain struct passed `no_diff`, so it has to become
  `IDifferentiable` first. Clamp `G` at 10, per Mitsuba and per the predecessor's §9.
- **Throughput.** The estimator must be scaled by `irradianceThroughput`, which a
  standalone post-resolve pass never sees — see §4.

This is the bulk of the remaining work.

---

## 7. Gap G — estimator and weighting

Staging is as the predecessor sets it out: biased budget-and-dedupe first (its §9), then
the Bernoulli-trial unbiased estimator (§10.1), then MIS against the existing NEE and BSDF
techniques (§10.2). Two local interactions it does not mention:

- **The firefly clamp will eat the caustic.** `fireflyClampLuminance` exists precisely to
  suppress the high-variance paths this feature is built to find; its own comment at
  `app/viewer/shaders/shared/global_pc.h:133` names caustics as the case it is clamping.
  Decide whether manifold contributions are exempt from the clamp or whether the clamp
  rises when the feature is on. Leaving it alone means building the feature and then
  discarding its output.
- **The valid-sample spp convention.** `irradiance.a = float(hit)` means a path whose only
  light arrives through a caustic connection changes what counts toward spp. Settle this
  before trusting any before/after comparison made through the viewer, since that
  convention already breaks estimator A/Bs done that way.

---

## 8. Gap H — loose wiring to finish or delete

- `manifoldNeeEnabled` is pushed from `app/viewer/main.cpp:1334` into
  `app/viewer/shaders/shared/global_pc.h:91` and
  `lib/ponder/shaders/util-global-pc.slang:64`, and is **read by no shader**. It is the
  flag the integrated version branches on; today it is declared and dead.
- `manifold-shade.slang` and `GpuManifoldShadePC` are superseded by the inline decision in
  §4 and should be deleted once the inline path lights a pixel. Keep them until then —
  they are the only current way to see where the solver lands.
- The counts buffer (`main.cpp:521`), the resolution-sized bend-point buffer (`:812`) and
  the ImGui readout (`:3050`) are probe scaffolding. Decide which survive as diagnostics
  once real contributions land.
- `stage::LightInstanceQuad` still reaches no consumer (predecessor §6.4, unchanged). Not
  blocking: glTF emissive geometry works through the existing emitter list.

---

## 9. Gap I — validation

Better placed than the predecessor assumed. `unit-test/src/test-dielectric-scenes.cpp:1285`
already renders the glass-ball caustic, prints the floor p99.9-to-median luminance ratio as
the instrument, and its `--thorough` 65536-spp run is exactly the brute-force reference the
predecessor's §13 asks for.

Two caveats:

- It is the two-bounce case, which §1 now targets directly — so it is usable as-is, and no
  new scene has to be authored.

### 9.1 Baseline, feature off — measured 2026-10-28

The step-0 before-numbers, from `vkof-test --test-case="caustic: glass ball lens over floor"`
on `build/release`:

| run | spp | floor median | floor p99.9 | ratio |
|---|---|---|---|---|
| default (smoke) | 512 | 0.172874 | 5.25703 | **30.4096** |
| `--thorough` | 65536 | 0.151646 | 5.09347 | **33.5879** |

Reference image at `unit-test/output/caustic-glass-ball-lens.png`; the caustic focus under
the ball is already visible at 65536 spp, which is what makes this a usable reference
rather than just a smoke test.

Read the ratio, not the median: between 512 and 65536 spp the median *falls* 12% while the
ratio *rises* 10%. The brute-force estimator is still filling in the tail at 512 spp, so
only the `--thorough` row is a fair comparison target. A working manifold sampler should
reach something near the 33.6 ratio at far fewer samples; matching 30.4 at 512 spp proves
nothing on its own.

The second step-0 number — the tile budget's per-tile gpu-ms (§4) — still has to be taken
from the windowed viewer, which needs a display and so cannot be captured from a headless
session.
- The existing unit tests (`unit-test/src/test-slang.cpp:35,75`) cover the constraint and
  the solver only. Nothing covers the contribution or `G`. `G` in particular is a Jacobian
  determinant and deserves the same finite-difference check the constraint already gets in
  `unit-test/shaders/slang/manifold.slang:97`.

Heavy render sweeps stay behind `--thorough` via the `test::statistical_*` helpers.

---

## 10. Suggested order

§1 and §4 are decided (chain of two; inline in the NEE block), so the order is now fixed.

0. **Baseline the instrument.** Record the glass-ball scene's floor peak-to-median ratio
   and the tile budget's per-tile gpu-ms with the feature off. Both later steps are judged
   against these two numbers, and neither can be recovered afterwards.
1. **§2 and §3** — correct caster addressing and a caster flag. Until these land the
   convergence rate is not measuring the right thing, so no later tuning can be trusted.
   §3 is also one of the two gates the inline decision depends on.
2. **§1a** — the chain-of-two solver, against the existing unit tests, extended to refract
   twice. No renderer involved; this stage finishes when a known two-interface path solves
   back to its own bend points.
3. **§4 and §5** — move seeding to a real path vertex with a real emitter sample, behind
   `manifoldNeeEnabled`. Take the divergence measurement here, while the contribution is
   still zero and the cost is attributable.
4. **§6's `G`** — return the Jacobian from `solve`, make `Path` differentiable, add the
   determinant and its clamp, with a finite-difference test beside the existing one.
5. **The rest of §6** — biased contribution plus the visibility rays. First caustic pixels
   appear here.
6. **§7's clamp and spp decisions, then MIS.**

Steps 1, 2 and 4 are where real bugs are most likely; the rest is plumbing.

---

## 11. House rules that apply

- Bracket edited chunks in `lib/` and `app/` with `(TODO REVIEW)` before and after.
  `unit-test/` is exempt.
- Comments one to three lines, always on their own line, never trailing.
- Build and test against `build/release`.
- Default test runs stay light on the GPU; heavy sweeps go behind `--thorough`.
- `lib/` code stands alone — no cull-port commentary.
