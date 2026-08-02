# Media stack — implementation plan

Scope: participating media in the wavefront core (`lib/ponder/shaders/`). Volumes are
bounded by closed meshes; a per-path medium stack tracks which media a path is inside;
`traceTransport` samples scatter distances through the active media on every chord
(flat or curved). Light sampling inside media (NEE, equiangular, ReSTIR) is **not** in
scope, but the interfaces leave room for it (§7).

---

## 1. What already exists

| piece | where | reuse |
|---|---|---|
| `TransportEvent::MediumScatter` | `path/transport.slang` | declared, never produced |
| emission-with-transmittance continuation | `entry/wavefront-integrate.slang` (Emission case) | template for crossing a boundary without a bounce |
| per-chord transport loop | `path/transport-schwarzschild.slang` | sample media per chord |
| Draine / HG / Rayleigh / Cornette–Shanks phase | `shaders-old/util-vdb.slang` (`utilVdbPhaseDraine`) | port into `I_PhaseFunction` |
| Jendersie–d'Eon fog phase by droplet diameter | `shaders-old/util-vdb.slang` (`utilVdbPhaseFog*`) | port |
| homogeneous closed-form distance sampling | `shaders-old/util-fog.slang` | port |
| subsurface random walk, Dwivedi guiding, albedo→σ, channel sampling | `shaders-old/util-material-openpbr-subsurface.slang` | becomes a medium behind the same interface (§6 phase 4) |
| per-volume parameters | `stage::VdbBlob` (`lib/stage/include/stage/stage.hpp`) | field list for the medium material extension (VDB itself is being dropped) |
| path debugger, timeline, dimension check | `types/debug-path.slang`, `ponder/debug-path.hpp` | scatter vertices, replay, sampler-slot reuse |

---

## 2. Assumptions (stated in the interface)

- Exponential transmittance (Beer–Lambert).
- Elastic scattering: no wavelength change, so the 4 hero wavelengths stay independent.
- No polarization.

Out of scope: fluorescence / Raman, non-exponential (correlated) media, continuously
varying IOR (that is a curved-transport problem, not a medium one).

---

## 3. Design

### 3.1 Phase functions

```slang
public struct PhaseSample { f32v3 wi; f32 value; f32 pdf; }
public interface I_PhaseFunction {
	func evaluate(wo: f32v3, wi: f32v3) -> f32;
	func sample(wo: f32v3, xi: f32v2) -> PhaseSample;
}
```

Implementations: isotropic, Henyey–Greenstein, Draine (covers Rayleigh and
Cornette–Shanks), fog mixture (Jendersie–d'Eon), two-lobe HG. `PhaseFunction` is a tagged
struct that switches over them (GPU dispatch is a switch).

### 3.2 Medium

```slang
public struct MediumPoint {
	Spectrum sigmaA;
	Spectrum sigmaS;
	Spectrum emission;      // zero for non-emissive media
	PhaseFunction phase;
}

public struct MediumSegmentSample {
	bool scattered;         // false: reached the end of the segment
	f32 distance;
	Spectrum weight;        // throughput factor, spectral MIS included
	MediumPoint point;      // valid when scattered
}

public interface I_Medium {
	func at(x: f32v3, wl: Wavelengths) -> MediumPoint;
	func sampleSegment(
		ori: f32v3, dir: f32v3, length: f32,
		wl: Wavelengths, random: MediumRandom
	) -> MediumSegmentSample;
	// upper bound on σₜ along the segment; σₜ itself for homogeneous media
	func majorant(ori: f32v3, dir: f32v3, length: f32, wl: Wavelengths) -> Spectrum;
}
```

`majorant` is exposed so tracked paths have an evaluable pdf (the null-collision path
integral, Miller et al. 2019): ReSTIR in volumes, spectral mis in chromatic media and
differentiable volumes all need it. `sampleSegment` still returns one real event; null
collisions stay internal.

Kinds:

- **homogeneous**: closed-form free flight.
- **height fog**: σ(h) = σ₀·exp(−(h − h₀)/H); optical depth along a straight segment is
  closed-form, so distance sampling is exact.
- **density field**: σ = density(x) × base coefficients; null-scattering (delta / ratio
  tracking) against majorant segments from a coarse min/max grid. Fields are procedural
  (`I_DensityField`: analytic, SDF shell, noise, composition) or baked to the native brick
  grid; no VDB.

Coefficients are spectral at the path's wavelengths (RGB uplift; measured spectra later).

### 3.3 Medium stack

Each path carries a small fixed stack (4 entries) of `{medium, priority}`.

- **priority > 0: exclusive.** Only the highest-priority exclusive medium applies (water
  inside glass). Schmidt & Budge 2002 nested dielectrics.
- **priority 0: additive.** Coexists with everything else (smoke inside haze).
- **entry 0** is the scene haze (or vacuum), never popped.

The active medium set is the top exclusive medium plus every additive medium. With more
than one active medium:

- σa and σs add;
- the scattering medium is chosen in proportion to σs, and its phase function is used;
- density-field majorants add per segment.

### 3.4 Boundaries

- A **medium boundary** material: closed mesh, no BSDF. The path crosses it without
  counting a bounce (like the disk's transmittance continuation).
- An AABB volume is a scaled box mesh with a boundary material; no separate AABB path.
- Front-face hit: enter the material's medium. Back-face hit: exit it.
- Iray rules for coincident or redundant boundaries (Ray Tracing Gems ch. 11): ignore a
  front-face hit for a medium already on the stack, and a back-face hit for one that is
  not.

### 3.5 Distance sampling in the transport

- The transport owns the `MediumStack`, like its origin, direction and time.
- Every chord calls `sampleSegment` on the active media. Free flight is memoryless, so each
  chord samples independently; nothing is carried across chords. Curved transport needs
  no special handling.
- Distances are sampled with the hero wavelength's σt; the other wavelengths are weighted
  by spectral MIS (pbrt-v4, Wilkie et al. 2014). Spectral / decomposition tracking (Kutz
  et al. 2017) is the reference for chromatic density fields.
- A scatter before the surface returns `MediumScatter` with the `MediumPoint`; otherwise
  the surface hit is returned and throughput is multiplied by transmittance.

### 3.6 Integrator

- `Surface` with a boundary material: `transport.crossBoundary(medium, priority,
  frontFace)`, loop without counting a bounce. The transport stays unaware of materials.
- `MediumScatter`: multiply in the weight, add emission, sample the phase function,
  `newDirection`. Each scatter counts as a bounce.

### 3.7 Data

- Medium properties are a **material extension** (a medium describes a volume, not a
  surface). Fields from `stage::VdbBlob`: σa, σs (RGB), phase type and parameters (droplet
  diameter, anisotropy override), emission and temperature scale, density field
  reference, priority, `isMediumBoundary`.
- Duplicating a material per medium is accepted.
- Global gets the scene haze parameters (or a medium index for entry 0).

### 3.8 Sampler and debugging

- New `BounceDimension` entries: `MediumDistance`, `MediumSelect`, `PhaseDirectionX`,
  `PhaseDirectionY`.
- `MediumRandom`: delta tracking needs an unbounded number of random numbers; a small
  stream seeded from `MediumDistance` with an internal counter, so the dimension check
  stays meaningful.
- Debug path: scatter vertices record the medium index and the medium stack.

---

## 4. Majorants

The weak point of null-scattering: a loose majorant means many null collisions; an
underestimate biases the image. Ratio and weighted delta tracking get extreme variance
with a bad majorant (Misso et al. 2023).

- Density fields bake a coarse min/max grid (RenderMan style), walked as majorant
  segments; this fits the planned baked brick grids.
- Additive overlaps sum majorants per segment.
- Optional density clamp to a chosen majorant (bias for stability), off by default and
  off in reference mode.

---

## 5. Validation

- **Homogeneous slab**: transmittance and single-scattering against closed form.
- **Furnace**: albedo-1 medium inside a white environment converges to the environment.
- **Height fog**: analytic optical depth vs a ray-marched reference.
- **Overlap**: two additive homogeneous media equal one medium with summed coefficients;
  exclusive priority equals the top medium alone.
- **Nested boundaries**: water in glass in haze; Iray rules prevent leaks on coincident
  faces.
- **Debug path**: scatter vertices, no dimension reuse across scatters.
- **Validity overlay** (pending instrumentation): no NaN or negative XYZ.

---

## 6. Phases

1. **Interfaces and scene haze.** `I_PhaseFunction` (port the old phase functions),
   `I_Medium`, `MediumStack` with only entry 0, homogeneous and height-fog haze,
   `MediumScatter`, sampler dimensions, debug vertices. Works through the curved transport.
2. **Boundary meshes.** Medium material extension, boundary material, push/pop with
   priority and additive semantics, Iray rules, homogeneous media inside meshes.
3. **Density fields.** `I_DensityField` (procedural), baking to the brick grid, min/max
   grid, majorant segments, null-scattering, emission from temperature.
4. **Subsurface as a medium.** Port the random walk behind the same interface, with
   Dwivedi guiding as a distance and direction sampling strategy.

---

## 7. Later, outside this plan

- Light sampling at scatter points (NEE), equiangular sampling toward small lights (Kulla
  & Fajardo 2012), volumetric ReSTIR (Lin et al. 2021). The medium interface should allow
  a distance sampler other than free flight.
- Directional media (SGGX microflakes) as another `I_Medium`.
- Aggregate-to-volume LOD: distant grass or foliage baked to a brick grid with an
  anisotropic (microflake) phase function, fitted to the real geometry by autodiff
  (Loubet & Neyret 2017). Builds on phase 3 density fields plus SGGX.
- Measured spectral coefficients.
- Refractive-index gradients (heat haze, mirages) are a bending layer, not a medium. The
  gradient ∇η comes from forward-mode autodiff of the procedural field, with no finite
  differences; the same trick gives SDF normals.

---

## 8. References

- pbrt-v4, media and volume integrators: https://pbr-book.org/4ed/Volume_Scattering/Media.html,
  https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Volume_Scattering_Integrators.html
- Kutz et al. 2017, spectral and decomposition tracking:
  https://www.jannovak.info/publications/SDTracking/SDTracking.pdf
- Misso et al. 2023, progressive null-tracking:
  https://cs.dartmouth.edu/~wjarosz/publications/misso23progressive.pdf
- Pixar, production volume rendering:
  https://graphics.pixar.com/library/ProductionVolumeRendering/paper.pdf
- RenderMan volume aggregates: https://rmanwiki.pixar.com/x/bADbB
- Ray Tracing Gems ch. 11, nested volumes (Iray):
  https://link.springer.com/content/pdf/10.1007/978-1-4842-4427-2_11.pdf
- Cycles volume priority proposal: https://projects.blender.org/blender/blender/pulls/118478
- Kulla & Fajardo 2012, importance sampling in participating media:
  https://fpsunflower.github.io/ckulla/data/egsr_2012_volume_paper.pdf
