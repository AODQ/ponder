# Geometry at scale — notes

On Nanite-style virtualized geometry (cluster LOD hierarchies, NVIDIA's RTX Mega
Geometry) and what to do instead.

## Why skip mega geometry

A large engineering sink with little research angle, and a poor fit for this renderer.

- **Vendor lock:** the ray-traced version relies on NVIDIA's cluster acceleration
  structure extension. Without it, BLASes have to be rebuilt per LOD change by hand.
- **LOD is ill-defined here:** cluster LOD is chosen from camera distance, which is
  wrong under lensing and for secondary bounces. Mismatched LODs between rays also give
  cracks and self-shadowing.
- **It conflicts with the direction:** the decision was to own procedural content, and
  mega geometry is about ingesting huge scanned assets.

## Geometry-scale ideas that do fit (ranked)

1. **Aggregate-to-volume LOD.** Distant grass or foliage becomes a baked brick grid
   with an anisotropic (microflake) phase function, fitted to the real geometry by
   autodiff. Reuses `I_DensityField` and the media stack, and is still an open research
   area (Loubet & Neyret 2017 is the starting point).

2. **Implicit surfaces (SDFs) as first-class geometry.** Sphere tracing already works
   per chord, so it composes with bending transports. Autodiff gives the normals and
   the derivatives the manifold solver needs, where triangles only give interpolated
   normals.

3. **Tessellation-free displacement** (Thonat 2021). Ray-march a min-max mip hierarchy
   of the height map inside the base triangle's prism. Micro-detail without storing the
   triangles, and the height function is differentiable.

4. **Footprint-driven LOD.** The ray differentials through the transport stack (see
   `notes.md`) give a correct per-ray footprint under lensing. Use it to pick the
   displacement mip, the SDF step tolerance, or stochastic LOD between instance levels.

5. **Opacity micromaps.** Cross-vendor in Vulkan and the practical win for alpha-tested
   grass if the GPU supports them. Engineering, not research.

## Adopt early

Don't let the path vertex record assume triangles. "Hit prim + barycentrics" in
`notes.md` should be "geometry id + surface parameters", behind an interface that
differentiably reconstructs position and normal. Then SDFs and displaced surfaces fit
into the connection solver later without a rewrite.
