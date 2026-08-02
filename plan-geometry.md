# Geometry — plan

Scope: geometry beyond triangle meshes, owned and procedural rather than ingested at
scale. From `notes-2.md`. Mega geometry (cluster LOD, RTX Mega Geometry) is out of
scope: vendor-locked, its camera-distance LOD is wrong under lensing and for secondary
bounces, and it targets scanned assets rather than owned content.

---

## 1. What already exists

| piece | where | state |
|---|---|---|
| geometry-agnostic surface hit | `SurfaceHit` (`model/surface.slang`) | `kind` + `instance` + `element` + `parameters`; only `GeometryKind.Triangle` |
| reconstruction from a hit | `surfaceGeometryFromHit` | switches on `kind`; triangles in `implSurfaceGeometryTriangle` |
| procedural intersection hook | `path/trace.slang`, `CANDIDATE_PROCEDURAL_PRIMITIVE` | empty branch, "unsupported for now" |
| path vertex record | `PathVertex` | holds the `SurfaceHit`, so new kinds need no record changes |

A new geometry kind adds: a `GeometryKind` value, intersection in the procedural
branch, and a reconstruction case. Vertex record, consumers and the connection solver
stay unchanged.

---

## 2. Targets

1. **Implicit surfaces (SDFs) as first-class geometry.** AABB instances in the tlas,
   sphere tracing in the procedural branch. Composes with bending transports because
   it runs per chord. Normals and the derivatives the connection solver needs come from
   autodiff of the field instead of interpolated vertex normals.
2. **Tessellation-free displacement** (Thonat et al. 2021). Ray-march a min-max mip
   hierarchy of the height map inside the base triangle's prism, through the same
   procedural branch. Micro-detail without stored triangles; the height function is
   differentiable.
3. **Footprint-driven LOD.** Needs ray differentials through the transport stack (not
   planned yet). The per-ray footprint, correct under lensing, picks the displacement
   mip, the SDF step tolerance, or a stochastic LOD between instance levels.

Not planned: opacity micromaps. Engineering, worth doing when alpha-tested foliage
becomes a bottleneck.

---

## 3. Differentiable reconstruction

When the connection solver arrives, `surfaceGeometryFromHit` becomes differentiable in
the surface parameters: vertex-buffer loads stay `no_diff`, the barycentric (or field)
evaluation is differentiated, and it joins the autodiff check build
(`plan-path-infrastructure.md` §5).
