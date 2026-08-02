# Roadmap — order of work across the plans

Written 2026-11-14. Orders the open plans against each other; it does not replace them.
Each step names the plan that owns its detail. Where this file and
`plan-path-infrastructure.md` §6 disagree on order ("nee + mis, then ReSTIR"), this file
is newer.

The gaps that no plan covers yet (scene interchange, OpenUSD, output pipeline, light and
geometry types) are in `notes-3.md`; §4 below says where they slot in.

---

## 1. Main sequence — light transport

Every later step consumes two things: a light sample with a pdf, and a connection with a
Jacobian. Steps 1–5 build those and are cheap; do them in order.

| # | step | owner | depends on | done when |
|---|---|---|---|---|
| x | NEE + MIS at surface vertices, with tests | `plan-nee.md` tiers 1–2 (tests only; implementation sections are old-pipeline) | — | NEE-only, BSDF-only and MIS means agree on the rectangle-light closed form; `pdfLight` round-trips `sampleLight` |
| x | shadow-ray offset | `plan-shadow-ray-offset.md` | 1 | the three `0.001` literals in `path/transport-*.slang` and the `newDirection` hack are gone |
| x | NEE at medium scatter vertices | `plan-media-stack.md` §7 | 1 | furnace and single-scattering slab tests pass with NEE on |
| x | connection interface: transports return `TraceTransportConnection` | `notes.md` item 5 | 1 | `sampleLight`'s `dist² / cos` is the straight-line case of `areaJacobian`; image unchanged |
| x | environment map as a real light; native emission spectra | (none yet) | 1 | sky is importance-sampled and MIS-weighted; the list's environment entry is live |
| 6 | ReGIR, P0 measurement first | `plan-many-lights.md` | 1, 5 | P3 gate: beats the power cdf on bistro at equal time, or stop (2026-11-22: implemented through P3b; the gate measurement is not recorded) |
| 7 | caustic solver, biased, behind the connection interface | `plan-caustics-integration.md` steps 1–5 | 4, track A (dielectric), autodiff readiness | glass-ball caustic on screen |
| 8 | ReSTIR, direct light at the primary hit | (none yet) | 6, AOVs, per-pixel history | equal-time win over ReGIR alone on a many-light scene |
| 9 | unbiased caustics + MIS + reservoir reuse of solved paths; lensed NEE | `plan-slang-caustics.md` §10, `notes.md` | 7, 8 | caustic mean matches a long BSDF-only reference |
| 10 | path reuse (ReSTIR PT), volumes in ReSTIR, PSSMLT | `plan-path-infrastructure.md` §4 | 8, media phase 3 majorants | — |

Why this order:

- **ReGIR before ReSTIR.** ReGIR only replaces light selection, needs no screen-space
  buffers, and works at every bounce, in media and under lensing. ReSTIR then draws its
  candidates from it.
- **Biased caustics before ReSTIR.** The connection solver is the first research feature
  (`notes.md`). ReSTIR is the larger engineering lift and is what makes the unbiased
  solver's Bernoulli trials affordable, so the unbiased stage waits for it.
- **Connection interface early.** After ReGIR, ReSTIR and the solver are built on
  `visibilityQuery`, changing its shape costs three rewrites.

---

## 2. Parallel tracks

**A. Materials — OpenPBR over `I_Spectrum`.** One function at a time with the MaterialX
crosscheck. Start any time after step 1. Gates step 7 (needs a dielectric) and step 8
(the target function needs real bsdf eval and pdf). First use of `isDeltaDirac`.

**B. Content.**

- Media stack phases 2 → 3 → 4 (`plan-media-stack.md`). Phase 3 must expose majorants;
  step 10 needs them.
- Geometry (`plan-geometry.md`): SDFs, then tessellation-free displacement. SDFs before
  step 9: autodiff normals suit the solver better than interpolated ones.

**C. Infrastructure left from the rewrite list.** Per-instance velocity, cpu-target tests,
AOVs, unbiased reference mode, golden images. AOVs gate step 8 and the denoiser; golden
images should follow step 2, not precede it.

---

## 3. Not on the path

- `plan-graph-editor.md`: independent tooling. If wanted at all, before step 8, where
  the multi-pass reservoir buffers would be easier to wire with it.
- `plan-slang-port.md`: done.

---

## 4. Where the unplanned gaps slot in

Detail and sources in `notes-3.md`.

| gap | when | why then |
|---|---|---|
| renderer-side scene api (create / update / delete mesh, instance, material, light, camera) | before the second scene frontend, at the latest with step 5 | glTF via `mor` / `stage` becomes one frontend; USD another; edits become incremental |
| light types beyond emissive triangles and quads (sphere, disk, distant, dome, shaping) | with step 5 | all are `I_Light` cases; the delta ones exercise `isDeltaDirac` |
| thin-lens camera | any time | pinhole is the only `I_Camera` |
| texture lod (ray cones or differentials) | with track A | `lod` is `0.0f` everywhere in the new core |
| denoiser | after AOVs | needs albedo and normal AOVs |
| output: multi-layer exr, render settings, headless batch | after AOVs | prerequisite for any Hydra delegate |
| subdivision surfaces, curves | before OpenUSD content | USD meshes default to Catmull–Clark |
| OpenUSD | after track A, step 5, AOVs | a delegate with one material, one light type and no AOVs is not usable |
| paint-by-light (README capstone) | after step 9 | needs autodiff through the whole estimator and re-evaluation from a recorded path |

---

## 5. Plans that need a revision pass before use

| plan | stale part |
|---|---|
| `plan-nee.md` | implementation sections (old pipeline); test tiers still apply |
| `plan-many-lights.md` | revised 2026-11-16 (§R); unrevised prose maps through §2.1 |
| `plan-shadow-ray-offset.md` | list of literal sites |
| `plan-slang-caustics.md`, `plan-caustics-integration.md` | g-buffer receiver, probe and shade passes, `util-manifold*.slang` paths |
| `plan-path-infrastructure.md` | §6 order line only |
