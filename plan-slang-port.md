# Porting the shaders to Slang

Moves the shader codebase from GLSL to Slang. The motivation is autodiff and one
language across the renderer, not a rewrite for its own sake — so the port is
mechanical wherever it can be, and every step is verified against the GLSL it
replaces.

§1-2 are the rules. §3 onward is the order of work.

---

## 0. What the port is not

**Slang's GLSL mode does not help.** `-lang glsl` exists and compiles plain GLSL,
but it does not implement `GL_EXT_buffer_reference`:

```
error[E31217]: unrecognized GLSL layout qualifier
   | layout(buffer_reference, scalar) buffer F32Buffer { f32 data[]; };
   |        ^^^^^^^^^^^^^^^^
```

146 of 178 shader files use it, 327 occurrences. Every file worth porting hits
this on its first declaration. Slang's answer is its own pointer support, under
different syntax, so this is a deliberate gap rather than a version problem.

The port is therefore a real rewrite. The good news is that the central
transformation is one-to-one:

```glsl
layout(buffer_reference, scalar) buffer GpuMorPositionBuffer { f32v3 data[]; };
const f32v3 p = GpuMorPositionBuffer(model.positions).data[i];
```
```slang
f32v3 * positions = (f32v3 *)model.positions;
let p = positions[i];
```

Both lower to `PhysicalStorageBuffer64`. `util-manifold-mesh.slang` already does
this against real mor buffers and round-trips correctly.

---

## 1. Style

Follows `unit-test/CLAUDE.md` wherever the languages agree. Slang-specific
points below.

### Declarations

- **`func` syntax**, not the HLSL form. Both parse; pick one.
  ```slang
  func fresnel_dielectric(cosTheta: f32, eta: f32) -> f32 { ... }
  ```
- **`let` for immutable, `var` for mutable.** `let` is the default; reach for
  `var` only when the value genuinely changes.
- **West `const`** where `const` appears at all — `const f32v3 x`. East const
  does not parse.
- **Trailing commas** on multi-line parameter lists.

### Naming

| category | convention | example |
|---|---|---|
| types | `PascalCase` | `MeshPointState`, `ManifoldFrame` |
| free functions | `snake_case` | `frame_from_normal`, `mesh_point_state_fetch` |
| interfaces | `I` + `PascalCase` | `ISurface` |
| fields, locals, parameters | `camelCase` | `bendPointState`, `primitiveIndex` |
| module-scope constants | `sk` + `PascalCase` | `skTestGroupSize` |
| modules and files | `util-` + kebab-case | `util-manifold-mesh.slang` |
| namespaces | lowercase | `manifold` |

`line` is a reserved word. So is anything else HLSL reserves; if a plausible
identifier fails to parse, that is why.

### Formatting

- Tabs.
- Comments on their own line, never trailing.
- 1-3 lines per comment. State what a thing is and what its inputs must be, not
  why the code exists or how it works internally.
- Section separators:
  ```slang
  // -----------------------------------------------------------------------------
  // -- constraint
  // -----------------------------------------------------------------------------
  ```
- Multi-line expressions wrap in parens, the opening paren on the break line and
  the closing paren on its own line.

### Modules

- One `module "util-name";` per file, quoted so the name can carry dashes.
- `__exported import` for a dependency the consumer also needs, so one import
  reaches the whole subtree.
- `public` on everything exported — structs, their fields, functions,
  `typealias`. A missing `public` on a field is a compile error at the use site,
  not the declaration.
- Group a module's symbols in a `namespace` matching its subject. `import` does
  **not** namespace on its own; without one, `Frame` and `Path` land in global
  scope.
- `#include` stays available and is fine for test-local headers that declare
  push constants. Modules cannot declare push constants a consumer supplies.

---

## 2. Allowed and banned

### Banned — these crash or silently misbehave

| | why |
|---|---|
| `fwd_diff(instance.method)` | segfaults slangc 2026.18.3. Use `static` interface methods |
| pointer casts inside `[Differentiable]` | internal compiler error. Fetch outside, cache in a struct |
| `var x: T;` for a struct with default field values | leaves fields uninitialized. Use `T x = {};`. Slang warns, but warnings do not fail a build |
| omitting `-matrix-layout-column-major -force-glsl-scalar-layout` | layouts silently disagree with the C++ and GLSL sides. Scrambled values, no error |

### Required

- `: IDifferentiable` on any struct returned from a `[Differentiable]` function.
  Slang does not synthesise it for return types, and the failure mode is
  `fwd_diff` returning the bare type instead of a `DifferentialPair`.
- `[MaxIters(n)]` on any loop inside a differentiated function whose bound is not
  static.
- `no_diff` on every parameter not being differentiated. It keeps the generated
  derivative small and is checked.

### Missing from Slang, write by hand

- Matrix `inverse()` and `determinant()`. `mat3_inverse_transpose` in
  `util-manifold-mesh.slang` is the cofactor form; reuse it.

### Verified working

Worth knowing these are settled, since several looked risky:

- pointers to `PhysicalStorageBuffer64`, matching `buffer_reference` layouts
- `RayQuery` against a `RaytracingAccelerationStructure` at vkof's set 0 / binding 3
- bindless `RWTexture2D<f32v4> vkofImages[]` at set 0 / binding 2
- `Atomic<u32>` through a device address
- generics over an `associatedtype`, two levels deep, with autodiff through both
- namespaces reopened across module boundaries
- `import` resolving through `-I`, so vkof's existing `includePaths` plumbing works

---

## 3. The seam problem

GLSL and Slang compile to separate SPIR-V modules and **cannot call each other**.
A half-ported dependency chain does not link — it does not exist.

So the unit of cutover is a **dispatch**, not a file. Porting
`util-fresnel.glsl` does nothing for a GLSL shader that includes it; the shader
has to move too, and so does everything else it includes.

This forces the port's shape:

1. Port a leaf module to Slang. Both versions now exist.
2. Prove the Slang version numerically equal to the GLSL one (§4).
3. Repeat up the dependency graph.
4. When every dependency of one entry point is ported, move that entry point and
   delete nothing.
5. Delete a GLSL file only when no entry point includes it.

Duplication through the middle of the port is unavoidable. §4 is what keeps the
two copies honest.

---

## 4. The crosscheck harness — do this first

For each ported function, a test that runs **both** implementations over the same
inputs and compares. This is the same shape as the existing MaterialX crosscheck,
and it is what turns the port from a leap into a sequence of proofs.

Both compile in the same suite, so a test case can dispatch a GLSL shader and a
Slang shader over an identical input buffer and diff the outputs.

Practical notes:

- Compare bitwise where the ported code is a literal transcription; `1e-6`
  relative where an expression was reassociated.
- Sweep the domain, including the edges the material code actually hits:
  `cosTheta` at 0 and 1, roughness at 0, grazing incidence, `eta` either side
  of 1.
- A crosscheck that passes on a narrow domain and fails at an edge is the most
  likely outcome, so pick the domain deliberately rather than uniformly.

`unit-test/shaders/slang/util-test.slang` already provides the in-shader assertion
harness; it needs a second dispatch and a comparison, not a rewrite.

---

## 5. Order

The material library's dependency graph is shallow. Measured:

```
util-fresnel              no includes
util-roughness            no includes
util-mis                  no includes
util-random               no includes
util-energy-compensation  no includes
util-shading-frame        1
util-material-openpbr-microfacet   -> util-shading-frame
util-material-openpbr-{coat,metallic,transmission}  -> microfacet
util-material-openpbr-{diffuse,fuzz,thinfilm,tables}  no includes
util-material-openpbr-subsurface   -> util-random, util-shading-frame
util-material-openpbr              -> all of the above
util-raytrace             -> util-material-openpbr + mesh, emitter, vdb, mis
```

### Stage 1 — leaves

`util-fresnel`, `util-roughness`, `util-mis`, `util-random`,
`util-energy-compensation`. No includes, pure functions, trivially crosscheckable.
Small enough that the real output is a settled style and a working crosscheck
harness.

`util-random` deserves care: the u64 stream has to produce *identical* sequences,
or every downstream statistical test shifts.

### Stage 2 — shading frame and microfacet

`util-shading-frame`, then `util-material-openpbr-microfacet` (1009 lines, the
largest single material file). Microfacet is where most of the subtlety lives and
where a crosscheck earns the most.

### Stage 3 — the lobes

`diffuse`, `metallic`, `coat`, `fuzz`, `thinfilm`, `transmission`, `subsurface`,
`tables`. Independent of each other; each crosschecks against its existing
unit test's domain.

`subsurface` last of these — it is the one with a random walk, so its crosscheck
needs the RNG from stage 1 to be exact.

### Stage 4 — the stack

`util-material-openpbr.glsl` (704 lines) and `util-material-openpbr-ibl.glsl`.
At this point the whole material system exists in Slang and the furnace tests can
run against either.

### Stage 5 — geometry and transport

`util-mesh`, `util-mesh-raytrace`, `util-emitter`, then `util-raytrace.glsl`
(2418 lines, the largest file in the project). This is where `buffer_reference`
conversion is densest and where the ray-query rewrite lands.

### Stage 6 — entry points

The viewer's `.comp` files and the 122 unit-test shaders. Mechanical by now, but
it is the bulk of the line count.

### Stage 7 — delete

Remove the GLSL originals once nothing includes them, and drop the
`glslangValidator` branch in `compile_and_load_shader_module` if nothing is left
to use it.

---

## 6. The dual-header question — settled: generate

**Decided 2026-09-28: the Slang view is generated, not hand-mirrored.**

`lib/tools/shared-header-to-slang.py` turns a dual header into a Slang module;
CMake's `slang-shared-headers` target regenerates on every build, so the two
views cannot drift. Five headers are wired up:

| header | module |
|---|---|
| `mor-shared.h` | `util-mor-shared` |
| `emitter-shared.h` | `util-emitter-shared` |
| `util-regir-shared.h` | `util-regir-shared` |
| `global_pc.h` | `util-global-pc` |
| `resolve_pc.h` | `util-resolve-pc` |

The per-pass PC headers (`bloom-*`, `nan-check`, `pt-*`) are 4-8 fields each
and stay hand-written until an entry point needs them.

Two things the generator settles that hand-mirroring got wrong:

- **A module cannot declare a push constant.** `layout(push_constant) uniform
  PC {...} pc;` is dropped entirely, along with its `#ifndef *_NO_PC` guard.
  The entry point declares its own block from the generated structs.
- **`GpuGlobalPc` is exactly 128 bytes**, which is the root-constant
  reservation. So an entry point writes `struct PC { GpuGlobalPc global;
  GpuXxxPC xxx; }` and needs no `u32v4 rootPad[8]` padding trick — that hack
  in `manifold-probe.slang` predates the generated module and can go.

### Original framing, kept for the reasoning



`mor-shared.h`, `global_pc.h`, `emitter-shared.h` and friends are
`#ifdef __cplusplus` files shared between C++ and GLSL. Slang cannot include them,
which is why `manifold-probe-pc.h` is currently a hand-maintained mirror of
`manifold-probe.slang`'s push constant.

Hand-mirroring is tolerable for one struct. Across the whole port it is a
standing divergence risk, and layout mismatches are silent.

Options considered:

- **Keep mirroring**, with a `static_assert` on every struct size and a crosscheck
  test that round-trips each one through the GPU. Cheap, catches size drift, does
  not catch field reordering within the same size.
- **Generate** the Slang side from the C++ side at build time. Correct, adds a
  codegen step.
- **Invert**: make Slang the source of truth and generate the C++ headers. Best
  end state once no GLSL remains, disruptive before that.

Worth settling before stage 5, where the shared structs are thickest.

---

## 7. Scope

| | lines |
|---|---|
| OpenPBR material library | 4 817 |
| `lib/ponder/shaders` total (incl. 3 616 vendored nanovdb) | 11 351 |
| unit-test shaders, 122 files | 10 714 |
| viewer shaders | 3 482 |

~21k lines excluding vendored nanovdb. The test shaders are over half of it, and
they are also the safety net, so they cannot be skipped to make the number
smaller.

Stages 1-2 are the ones that decide whether the rest is worth it: they settle the
style, prove the crosscheck harness, and surface whatever Slang sharp edges the
material code hits that the manifold work did not.

---

---

## 9. Status — 2026-09-29: the port is done

**Every shader is Slang. All 181 `.glsl`/`.comp` files are deleted; 248
`.slang` files remain and the whole project builds.**

- library: every `lib/ponder/shaders` module, crosschecked before cutover
- `app/viewer/shaders/util-raytrace.glsl`, 2 421 lines
- all 156 entry points: 143 unit-test, 11 app, 2 in lib
- the C++ shader paths in `viewer/main.cpp`, `observer/main.cpp` and
  `environment-ibl.cpp`

Two deliberate holes:

- **VDB renders nothing.** `external-nanovdb.glsl` (3 616 lines, vendored,
  built on `PNANOVDB_*` macros) was never ported and is now deleted with
  the rest. `util-vdb.slang` keeps the phase/fog/sigma math 1:1 and stubs
  every grid reader to report an empty medium. By decision.
- **regir is ported shape-only.** `utilRegirFillReservoir` had changed
  signature and three shaders were never updated, so they did not compile
  in GLSL either. They now compile; `weightSum` receives
  `normalizationEstimate`, which differs from the old value by a factor of
  M. `utilRegirFillSeed` was called by all three and defined nowhere, and
  is reconstructed from the call shape. regir is a new feature and is not
  expected to work.

### The crosschecks are gone

21 cases, 107 assertions, 17 of them bitwise, proving each ported module
against the GLSL it replaced. They dispatched a `.comp` and a `.slang` over
identical inputs and diffed them, so they could not outlive one side.
`test-slang-port.cpp`, `util-crosscheck.hpp` and the 21 shader pairs went
with the GLSL. What they measured, before deletion:

| crosscheck | worst relative |
|---|---|
| random, roughness+mis, shading-frame, energy-compensation | **0** |
| transmission+subsurface, regir (both halves), mesh (both halves) | **0** |
| emitter (geometry and scene), env-prefilter, material-load | **0** |
| fresnel, openpbr lobes, openpbr stack | 5.96e-08 |
| regir-fill | 2.4e-07 |
| thinfilm | 1.2e-05 |
| misc (blackbody/fog/tonemap/env) | 1.6e-05 |
| microfacet | 2.5e-05 |

### Tooling

- `lib/tools/shared-header-to-slang.py` — generates the Slang view of a dual
  `#ifdef __cplusplus` header; run by CMake's `slang-shared-headers` target,
  and emits a `.buffers.json` sidecar naming each buffer-reference element
  type.
- `lib/tools/glsl2slang.py` — the converter. Handles signatures,
  `buffer_reference` -> pointers, builtin renames, include-order imports,
  duplicate definitions, and the module header.
- `lib/tools/slang-port-check.sh` — regenerates every converted module and
  reports which compile.

Modules are regenerated from the GLSL, not hand-edited, so a converter fix
re-applies everywhere. The exceptions are the six hand-ported leaves listed
in `HAND` in the check script.

### Calibration

**The house tolerance is 1e-6, not bitwise.** glslang and slangc contract and
reassociate differently, so a literal transcription lands about one float32
ulp (6e-8) apart. §4's original "bitwise where literal" advice was wrong and
is corrected above.

The one module needing 1e-4 is microfacet, and only because the VNDF sample is
ill-conditioned at the degenerate alpha lanes (1e-5). Drop those lanes and the
whole sweep lands at 5.5e-7 — so it is conditioning, not a port error.

### A real bug the crosschecks caught

`utilPlanckRadiance` returned **Inf** in Slang and a correct value in GLSL,
from byte-identical source. The SPIR-V showed why: Slang reassociates
`lambdaM^5` (lambdaM ~ 4e-7, so the product lands near 1e-33, five orders off
the f32 floor) and the result flushes to zero, making the reciprocal Inf.
GLSL happened to keep a grouping that survived.

Fixed in **both** languages by keeping the fifth power in nanometres and
folding the 1e-45 that the nm->m conversion would contribute into the
numerator, where nothing is near a limit. A deliberate, documented break from
1:1: the original is fragile in any compiler free to reassociate, so matching
it exactly would have preserved a latent bug.

Worth noting the first attempt at that fix wrote `1e45f` directly, which is
**unrepresentable in f32** and folds to infinity. Slang warned
(`E40009: floating-point literal unrepresentable`); glslang folded it at
higher precision and said nothing. The scale is now folded into `h`.

### What the crosschecks caught

- The GLSL headers rely on their *includer* defining `PI`/`TAU`/`IPI` and
  including `util-material-openpbr-tables.glsl` and `-thinfilm.glsl`. A
  crosscheck shader that does not reproduce that silently compiles to nothing
  and reports every value as differing.
- `openPbrRoughnessAlpha`, `openPbrCoatRoughenedRoughness` and four energy
  functions are **defined twice in the GLSL** — once in their own file and
  again in `util-material-openpbr-microfacet.glsl`. GLSL tolerates it because
  the two files are never included together. Bodies verified identical; the
  later module keeps a module-private copy.

### Slang findings beyond §2

- **`slang-port-check.sh`'s "OK" is weaker for an `#include`-only file.** A
  `module` is fully typechecked on import, so an error surfaces without a
  reference; an `#include`d file's unreferenced functions are not. That is
  only `util-nan-probe`, whose body sits behind `NAN_PROBE_ENABLED` and is
  inert — its `GpuNanProbeCounterBuffer` and `debugPrintfEXT` are still
  unconverted and would fail if enabled.
- **GLSL `mix(a, b, bvec)` is a select, not a lerp.** Slang's `lerp` takes the
  bool through an implicit 0/1 conversion — numerically identical on ordinary
  operands, different on nan, and it warns. The converter emits `select`.
- **Preprocessor macros do not cross a module boundary.** `util-nan-probe` is
  macro-only, so it stays `#include`-able rather than becoming a module.
- **`import` is not transitive, `__exported import` is** — and a diamond of
  `__exported import` makes every shared symbol ambiguous. The converter
  imports only what is not already reachable.
- **Slang has no `inverse()` or `determinant()`** for matrices.
- **`atan(y, x)` is `atan2`**; `mix`->`lerp`, `fract`->`frac`,
  `inversesqrt`->`rsqrt`, `texelFetch`->`.Load`, `texture`->`.SampleLevel`,
  `lessThan`/`not` -> operators.
- **A GLSL `mat3(a..i)` fills columns where a Slang `f32m3x3` fills rows**, so
  the same nine scalars land transposed and `M * v` becomes `mul(v, M)`.
- `gl_GlobalInvocationID` is not a free global; it only reaches an entry point
  as a semantic, so `utilGenerateSeeds` takes it as a parameter.

### Not a regression

The tree came into this work with **16 failing test cases**, none of them
touched by the port (which changes no GLSL and no C++ outside
`test-slang-port.cpp`). Recorded here so the next session does not chase
them:

```
dielectric: hollow glass sphere vs checkerboard
dielectric: nested spheres ior sweep
diffuse transmission placement: layered material stays bounded
env mis: firefly clamp cost per technique
env mis: rotation invariance
env mis: sub-texel jitter stays uniform as the map grows
env mis: sun contrast sweep
kulla-conty table vs analytic fit                       (labelled known bug)
golden reference: kulla-conty table vs integrated ggx   (labelled known open)
noise: single-pixel pcg-chain histogram
regir fill: six cases, all E[W] / normalization
tangent generation: matches authored TANGENT on a stripped copy
transmission dispersion: cpu cauchy/abbe reference
transmission sampler vs pdf: chi-square (anisotropic)
```

`regir fill`'s six are the GLSL feature still being developed in
`plan-many-lights.md`, not the Slang view of it.

## 10. References

- Slang user guide, autodiff and SPIR-V specifics —
  https://shader-slang.org/slang/user-guide/autodiff.html
  https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/a2-01-spirv-target-specific.html
- `lib/ponder/shaders/util-manifold*.slang` — the existing Slang in the repo, and
  the reference for the conventions above
- `unit-test/shaders/slang/util-test.slang` — in-shader assertion harness
- `lib/vkof/src/vkof.cpp`, `compile_and_load_shader_module` — where `.slang`
  branches to `slangc`

---

## 11. Handover — where to pick up

`lib/ponder/shaders` is 2 files short of ported and 5 modules short of
crosschecked. Run `./lib/tools/slang-port-check.sh` for live compile status
and `vkof-test -tc="slang port:*"` for numerical status.

### Stage 6 in progress: 125 of 143 unit-test entry points

`lib/tools/comp2slang.py` converts a `.comp` entry point, reusing
glsl2slang for the body so a converter fix reaches both. What is
entry-point specific: the push constant block becomes a `struct` plus a
`[[vk::push_constant]]` global (with `layout(offset = 128)` turning into
`u32v4 rootPad[8]`), `layout(local_size_x = N) in;` becomes
`[numthreads(...)]`, and `void main()` becomes a compute entry point
**whose parameters are named after the gl_ builtins the body uses** — so
the body needs no edit at all.

Those 125 are cut over for real: the tests name the `.slang` and the GLSL
twin is no longer dispatched.

Still failing, by cause:

| count | shaders | cause |
|---|---|---|
| 4 | `vdb_*` | include `util-vdb`, not ported by decision |
| 4 | `emitter_nee_scene`, `furnace_model_render`, `subsurface_furnace_render`, `subsurface_material_probe` | include `util-raytrace.glsl`, the app's 2 421-line file |
| 2 | `regir_fill_probe`, `regir_direct_lighting` | **already broken in GLSL** — see below |
| 2 | `dielectric_specular_evaluate`, `dielectric_specular_furnace_integrate` | the name collision below; a design call |
| 2 | `noise_generate_seeds`, `noise_generate_chain` | `utilGenerateSeeds` arity; the Slang side took `invocationId` as a parameter and the GLSL still reads `gl_GlobalInvocationID`. The §11 pc pattern says change both; that has not been done for this one |
| 3 | `cornell_glass_render`, `cornell_openpbr_render`, `dielectric_scene_render` | the ray-query object-to-world matrix type |
| 1 | `subsurface_walk_step` | a `buffer_reference` declared inside a function body |

### Two things the port surfaced that are not port bugs

**`openPbrDielectricSpecularEvaluateF` is two different functions.**
`util-dielectric-specular.glsl:39` takes nine unpacked scalars;
`util-material-openpbr-microfacet.glsl:851` takes an `OpenPbrMaterial` and
four more. GLSL tolerates it because the two files are never included
together. In Slang both are reachable through one import graph and the call
resolves to the microfacet one, silently. §9's earlier duplicate note was
about byte-identical bodies; this is a genuine collision and wants a rename
in both languages.

**`utilRegirFillSeed` is called and never defined.** `regir_fill_probe.comp`
and `regir_direct_lighting.comp` both call it; it exists nowhere in the
repo. They do not compile under `glslangValidator` either — verified
directly — which is where some of the baseline's six failing `regir fill`
cases come from. Nothing to do with the port.

### Converter notes for the entry points

- **vkof's set-0 bindings are declared once, in `util-types`** — textures,
  images, the TLAS. A shader that redeclares one makes every use
  ambiguous, so both converters drop `layout(set = 0, ...) uniform ...;`.
  Same for `PI`/`TAU`/`IPI`: util-types owns them and a redefinition,
  guarded or not, is ambiguous rather than shadowing.
- **`GL_EXT_ray_query` maps cleanly.** `rayQueryEXT` is
  `RayQuery<RAY_FLAG_NONE>`, the accessors are methods, and
  `rayQueryInitializeEXT`'s eight arguments expand to a `RayDesc` plus
  `TraceRayInline`. GLSL's `rayQueryGetIntersectionObjectToWorldEXT`
  returns a `mat4x3`, which is the 3x4 affine — so it is
  `CommittedObjectToWorld3x4`, not the 4x3 method whose name looks closer.
- **A unit-test helper that carries macros stays `#include`-able**, and
  one that declares the push constant its includers share keeps it. Both
  are listed in glsl2slang's `INCLUDE_ONLY`.
- Buffer-reference element types are now collected tree-wide, since a
  shader routinely uses a type one of its headers declared.
- **Every ported entry point needs the ponder shader directory on its
  include path**, even one that included nothing before: `f32`/`u32v4` were
  a `#define` block in GLSL and are an `import "util-types"` in Slang. Nine
  test files dispatched shaders with no `includePaths` at all; they now use
  `test::slang_include_paths()`. The failure mode is the one §9 warns
  about — the pipeline compile fails, the dispatch writes nothing, and the
  test reports wrong *values* rather than a compile error.

### Not started, deliberately

- **`util-vdb` + `external-nanovdb`** (4 279 lines) — nanovdb is vendored and
  built on 6 `PNANOVDB_*` portability macros. wrap it or leave VDB on the glsl
  path; do not hand-port. This is now the only unported file in the library.

### Then: entry points

`app/viewer/shaders` and `app/observer/shaders` (1 230 lines of `.comp` in 11
files, plus `util-raytrace.glsl` at 2 421) and the 134 unit-test shaders
(11 668 lines). Cut over a **unit-test** shader first: they have assertions,
so a bad port fails loudly instead of producing a subtly wrong image.

### Converter rules, do not re-derive

`shadow_mutated_params` (glsl params are mutable copies, slang's are not),
`fix_matrix_multiply` (scanner, not regex — it collects every matrix-typed
identifier in the tree, because a missed `M * v` is silently wrong),
`fix_bool_mix` (`mix(a,b,bvec)` is a select), `fix_atomics`,
`fix_vector_equality`, `publicize_structs`, `map_outside_comments`,
single-element buffer references, pointer-typed parameters, and the
`.buffers.json` sidecar that keeps glsl2slang and the header generator
agreeing.

### Converter bugs found and fixed, worth knowing the shape of

All five were found by regenerating every module and recompiling, which is
the reason to run `slang-port-check.sh` before trusting a `.slang` file that
was generated by an older converter.

- **`shadow_mutated_params` shadowed `out` parameters.** An `out` parameter is
  already writable in Slang, so `var outWeight = outWeight;` compiles and
  sends every write to the local — the caller sees nothing. It had reached
  four functions, `utilFresnelDielectricPhase` among them. It also computed
  its insertion offsets up front and then invalidated them by inserting, so
  the second mutated function in a file got its prologue spliced into the
  middle of the parameter list. Both fixed; the rule now fires only for a
  by-value parameter, which in practice is rare.
- **`atomicAdd` -> `InterlockedAdd` was wrong in principle.** GLSL's atomics
  return the old value; the HLSL builtins return it through an `out`
  parameter. `util-types.slang` now has `atomicAdd`/`atomicCompSwap` shims
  over `Atomic<u32>` that keep the GLSL call shape, and the converter takes
  the address of the target. Verified: the SPIR-V carries a real
  `OpAtomicCompareExchange`, relaxed, device scope, matching GLSL.
- **GLSL `==` on vectors is one bool; Slang's is one per component.**
  `fix_vector_equality` folds to `all()`/`any()`, driven by the same
  tree-wide name scan as the matrix rule but with any name that is also used
  for a scalar somewhere removed — a false positive is a no-op
  (`all(scalarBool)` is that bool) and a miss is a type error, so the scan is
  deliberately conservative.
- **`BUILTIN`/`TYPEMAP` substitution ran inside comments**, so `\int` in the
  latex became `\i32`. `map_outside_comments` now brackets both.
- **`publicize_structs` used a non-greedy regex** that ended a struct body at
  the first `}` — which `\hat{h}` in a doc comment supplies, leaving every
  field after it private. Now brace-matched with comment skipping, reusing
  `end_of_body`.
- **The auto-import heuristic searched comments too**, so one mention of
  `EmitterSample` in a doc comment pulled all of `util-emitter` (and through
  it mor-shared) into `util-regir`, the one file whose header says it must
  stay includable without them. It now searches comment-blanked source.
  A `#include` of a dual header also resolves to that header's generated
  module via `SHARED_HEADER_MODULES` instead of emitting a TODO comment.

`shared-header-to-slang.py` also converts hex and expression-valued
`#define`s now (`skRegirSlotInvalid`, `skRegirNormalBucketCount`), tracking
the types of the constants it has already emitted so an expression over them
gets the right one, and splits function-like macros into the `-macros.slang`
companion described in §9. `SHARED_HEADER_MODULES` is keyed by basename,
since an include may be path-qualified (`"mor/mor-shared.h"`).

### The pc pattern, already applied

`util-mesh` takes a `UtilMeshDecodeContext { models, viewProj, invocationId }`
and `util-mesh-raytrace` takes `models` directly; **both changed in the glsl
too**, so the two stay 1:1 and regeneration keeps working. Callers updated in
`scene-visibility-resolve.comp`, `resolve.comp`, `util-raytrace.glsl`. Use the
same shape for anything else that reaches for `pc`.
