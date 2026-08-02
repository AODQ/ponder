# Gaps — what the renderer is missing, and OpenUSD

Written 2026-11-14 from an audit of the tree plus a web pass on the current state of
OpenUSD, MaterialX and denoising. Companion to `plan-roadmap.md`, which says when each
item should happen. Claims marked *(unverified)* are from memory or a single secondary
source and should be checked before they drive a decision.

---

## 1. What exists (audit)

| area | state |
|---|---|
| scene | glTF through `mor`; `stage` holds instances, material overrides, quad lights, hdri environment, vdb and alembic instance records |
| camera | `CameraPinhole` only, with shutter time |
| lights (new core) | emissive triangles via the shared emitter list; sky is a placeholder checkerboard |
| materials (new core) | Lambertian; OpenPBR lives in `shaders-old` |
| transport | flat, Schwarzschild / Kerr, participating media (homogeneous, height fog) |
| colour | hero-wavelength spectral, Jakob–Hanika uplift, ACES tonemap |
| output | tinyexr is linked; no AOVs in the new core |
| tooling | path inspector, timeline, dimension-reuse check, autodiff check build |

---

## 2. Gaps, ranked by what they block

### 2.1 Blocks ordinary scenes

- **Materials.** Only Lambertian in the new core. Everything specular, and therefore the
  caustics work, waits on the OpenPBR port.
- **Texture lod.** `lod` is passed as `0.0f` throughout. Ray cones are the cheap fix;
  ray differentials through the transport stack (`notes.md`) are the correct one under
  lensing.
- **Environment light.** The emitter list has an environment entry with real power but
  the sky is a checkerboard, so NEE can pick a light that is not there.
- **Light types.** Emissive triangles and stage quads only. Missing: sphere, disk,
  cylinder, distant, dome, and shaping (cone, IES). Point and distant lights are delta
  lights: no BSDF-side MIS.
- **Camera.** No thin lens, so no depth of field. Shutter exists; per-instance velocity
  (object motion blur) is still on the rewrite list.

### 2.2 Blocks using the output

- **AOVs.** None. Needed by the denoiser, ReSTIR, Cryptomatte-style mattes and any Hydra
  delegate.
- **Denoiser.** Open Image Denoise is the obvious choice. v2.5.0 can import external
  buffers from Vulkan; its external-semaphore import is CUDA and HIP only so far, and it
  does not share textures, only linear buffers. Needs albedo and normal AOVs.
- **Batch output.** No multi-layer EXR writer, render-settings object or headless
  "render N samples to file" path for the new core.
- **Colour management.** Tonemap is ACES only. A defined rendering colour space with a
  D65 white point matters for USD (§3.4); OpenColorIO is the usual integration.
- **Adaptive sampling, firefly policy.** No variance estimate per pixel; the reference
  mode on the rewrite list is the place to define what clamping is allowed.

### 2.3 Blocks production content

- **Subdivision surfaces.** USD meshes default to Catmull–Clark; most production assets
  look wrong as polygon cages. OpenSubdiv is the standard implementation.
- **Curves and points.** Hair, fur, grass as `BasisCurves`. No curve primitive or hair
  BSDF.
- **Animation.** No time-sampled transforms or deformation in the new core; skinning and
  blend shapes are absent.
- **Instancing at scale.** Point instancers with nested prototypes map to TLAS
  instancing; the emitter list and medium tables are indexed per instance and will need
  to follow.
- **Volumes.** VDB was dropped in favour of owned density fields. Imported volume assets
  would need an offline bake into the native brick grid.
- **Texture scale.** No UDIM, no out-of-core texture cache.

### 2.4 Wanted by the research direction

- **Path re-evaluation from a record** (`EvaluateFromRecord`), not yet built. The
  connection solver, ReSTIR shifts and the paint-by-light capstone all need it.
- **Ray differentials** through the transport stack.
- **Validation against an external spectral reference** (pbrt-v4 or Mitsuba 3) on a few
  fixed scenes. The MaterialX crosscheck covers materials only.
- **Wavefront queues.** The core is ray-query with persistent threads; vkof has no
  indirect dispatch yet. Matters once paths vary a lot in length (media, caustics).

---

## 3. OpenUSD

### 3.1 State of the ecosystem

- Hydra 2 (scene indices) is the default for UsdImaging since 25.11. The Hydra 1 scene
  delegate path was deprecated in 26.03; 26.11 is expected to be the last release with
  the switch back to it, and 27.02 removes the switch.
- That deprecation is about the application side (scene delegates). The announcement
  does not say the render delegate API is going away, and I found no replacement
  renderer-side API. *(unverified beyond that thread)*
- OpenPBR is supported through MaterialX: OpenUSD 25.02 added it, and 25.05 was the
  first release to default to MaterialX 1.39.
- v26.08 is the newest release I found. v26.03 added Gaussian splats.
- UsdLux defines a default white light of intensity 1 as 1 nit, says most consumers are
  RGB, and leaves RGB-to-spectrum uplift unspecified other than that it must round-trip
  under the rendering illuminant.

### 3.2 Three ways in, cheapest first

| route | cost | gives | does not give |
|---|---|---|---|
| **offline converter** USD → `stage` / `mor` (OpenUSD's Python, run outside the renderer) | small | USD assets on screen; renderer stays dependency-free | live edits, DCC integration |
| **TinyUSDZ** linked in as a loader | medium | no OpenUSD build; its Tydra layer converts to render-ready meshes and materials | composition is still marked experimental; no Hydra, so no host applications |
| **Hydra 2 render delegate** (`hdPonder`) | large | the renderer inside usdview, Houdini, Blender, Omniverse; incremental edits | — |

The delegate is large because of what it forces on the renderer, not because of the
plugin code:

- a full OpenUSD build as a dependency (TBB, and in practice MaterialX and OpenSubdiv);
- the renderer as a library with a scene-edit api: Hydra hands over prims to create,
  update and delete, and expects BLAS / TLAS and material updates to be incremental;
- AOV render buffers and render settings in Hydra's form;
- MaterialX networks, not just an OpenPBR parameter block: pattern nodes (image, mix,
  noise, primvar readers) in front of the closure, which means shader generation or a
  graph interpreter.

### 3.3 Reference projects

- **gatling**: a GPU path tracer exposed as a Hydra delegate with a standalone that
  reads USD. Handles OpenPBR through MaterialX and its MDL code generation backend. Lists
  UDIM, volumes and displacement as not implemented, and needs the MDL SDK to build.
  Closest in shape to what `hdPonder` would be.
- **moonshine**: another Vulkan ray tracer with a Hydra delegate. *(unverified)*
- **Typhoon**: a reference path tracer for OpenUSD interoperability, presented at
  OpenUSD Days 2026. I could not read the slides; worth reading for what it treats as the
  required schema set.

### 3.4 Where this renderer does not fit USD

- **Spectral data.** USD carries RGB. Emission and reflectance arrive as RGB in a named
  colour space and go through the existing Jakob–Hanika uplift. Measured spectra, the
  disk's blackbody and anything else native would need a custom applied schema.
- **Units.** `types/units.slang` fixes Y = 1 as display white and geometrised units for
  gravity; USD has `metersPerUnit` and 1 nit per unit intensity. One conversion at the
  import boundary.
- **Curved transport.** No schema for a black hole. A custom prim type plus an applied
  API schema; Hydra passes unknown prim types through to a delegate that asks for them.
  *(unverified)*
- **Owned procedural content.** Density fields and SDFs are code here, not assets. They
  would be referenced from USD by a custom schema, not expressed in it.

### 3.5 Recommendation

1. **Build the scene api first, with no USD in sight.** A thin layer between the app and
   `wavefront`: create / update / delete for mesh, instance, material, light, camera,
   volume, plus settings and AOV outputs. `mor` + `stage` become its first client. This
   is useful on its own (incremental edits, the stage editor) and is exactly what a
   delegate needs.
2. **Take the offline-converter route when the first USD asset is needed.** It exposes
   the content gaps of §2.3 (subdivision, light types, cameras) without committing to the
   OpenUSD build.
3. **Write the delegate only when a host application is the goal.** By then materials,
   lights and AOVs must exist, or the delegate has nothing to show.

Skipping TinyUSDZ is deliberate: it avoids the build but not the content work, and the
converter covers the same need with less code in the renderer.

---

## 4. Sources

- Hydra 1 deprecation timeline: https://forum.aousd.org/t/deprecating-hydra-1-eventually/2804
- Hydra 2 getting started: https://openusd.org/dev/api/_page__hydra__getting__started__guide.html
- OpenUSD v26.03: https://aousd.org/blog/openusd-v26-03/
- OpenUSD v26.08: https://aousd.org/blog/announcing-openusd-v26-08-key-features-and-improvements/
- OpenUSD 25.02 and OpenPBR: https://80.lv/articles/alliance-for-openusd-launched-openusd-25-02
- MaterialX TAC update, Feb 2026: https://tac.aswf.io/meetings/2026-02-04/MaterialX_TAC_Update_2026_Final.pdf
- UsdLux overview: https://openusd.org/dev/user_guides/schemas/usdLux/overview.html
- UsdLux api front page: https://openusd.org/dev/api/usd_lux_page_front.html
- Typhoon slides: https://hosted-files.sched.co/osd2026/20/A%20Reference%20Renderer%20for%20OpenUSD%20Interoperability%20-%20Anders_Langlands.pdf
- gatling: https://github.com/pablode/gatling
- moonshine: https://github.com/ashpil/moonshine
- TinyUSDZ: https://www.github.com/syoyo/tinyusdz
- Open Image Denoise releases: https://github.com/RenderKit/oidn/releases
- Open Image Denoise documentation: https://openimagedenoise.github.io/documentation.html
- MoonRay feature overview: https://openmoonray.org/about
- appleseed feature list: https://appleseedhq.github.io/features.html
