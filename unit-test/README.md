# unit-test

Test binary (`vkof-test`, doctest) for `lib/ponder` and `lib/vkof`.

these are mostly AI driven tests

## OpenPBR port milestones

each lobe/term is ported from `cull` and independently verified (golden
values, a cpu f64 reference transcription, a materialx crosscheck ported
into `unit-test/shaders/`, nan-sweep heatmaps, and energy/furnace or
chi-square sampling checks where applicable) before being considered done.
completed: fresnel (dielectric + f82-tint conductor), ggx microfacet
distribution/visibility/directional albedo, eon diffuse + energy
compensation, glossy-diffuse layering, the metallic (conductor) lobe,
mixture-lobe sampling/pdf selection, fuzz (cloth) lobe, transmission
(dielectric btdf + dispersion), and **thin-film interference**
(`test-thinfilm.cpp`, landed 2026-07-16 -- see below). subsurface scattering
(`util-material-openpbr-subsurface.glsl`) is the one remaining unverified
term.

### thin-film interference (2026-07-16)

`openPbrThinfilmFresnel` (belcour & barla 2017) replaces the plain fresnel
term at the metallic and dielectric-specular interfaces with a wavelength-
aware interference color. verified byte-identical against cull's port (type
aliases aside) -- no lib code changed. coverage: golden values including a
zero-thickness-reduces-to-plain-fresnel identity, a full cpu f64 reference
transcription (worst gpu/cpu deviation ~0.006), a from-scratch materialx
`mx_fresnel_airy` port (`unit-test/shaders/util-thinfilm-materialx-
reference.glsl`) crosschecked on a dense grid, nan/bounds fuzzing including
the kappaBase != 0 (complex conductor base) branch that no current caller
actually reaches, and an energy-boundedness furnace check on the assembled
metallic lobe with thin-film active.

one open, bounded divergence from the materialx crosscheck: ponder/cull's
phase-shift term uses a sign-based simplification per polarization (`phi =
(r < 0) ? pi : 0`, blender cycles' simplification of the same belcour &
barla 2017 model) while materialx's `mx_fresnel_airy` instead gates both
polarizations off a single brewster-angle test. the two formulations agree
tightly near normal incidence (worst diff ~0.005) and diverge by up to
~0.11 (absolute, either side clamped to [0,1]) at grazing angles -- bounded
and understood, not a port bug, so the crosscheck test tolerates it with a
documented, measured epsilon rather than a tight one.

no chi-square/sampler test for this term: thin-film has no sampler of its
own (it only recolors the fresnel term inside an already-verified,
already-sampled ggx vndf lobe), and the metallic lobe's pdf function
(`utilMicrofacetGgxBoundedReflectPdfAniso`) doesn't take a material/fresnel
parameter at all -- it structurally cannot depend on thin-film by
construction, so a dedicated chi-square regression here would only re-
verify already-covered sampler code.

## RNG: u64 pcg32 streams everywhere (2026-07-16)

the openpbr sampling interface was migrated off cull's f32/f32v2 hash-chain
seeds (user decision, made while building the viewer): `openPbrSampleWo`
now takes one `inout u64 state` (a pcg32 stream from util-random.glsl) and
owns every draw; the leaf samplers (`openPbrFuzzSampleWo`,
`openPbrGlossyDiffuseSampleWo`, `utilCosineHemisphereSampleWo`) take a
`const vec2 xi` directly, matching the ggx samplers' existing convention.
this is a deliberate interface deviation from cull. consequences for tests:

- every `fnSampleUniform`/`fnSampleUniform2` f32-seed shim is gone (43
  shader files carried one), including the identity-shim technique the fuzz
  sampling suite used -- exact-xi injection is now just passing xi.
- callers of openPbrSampleWo build a stream as
  `u64 state = (u64(h0) << 32) | u64(h1)` from their existing pcg hash
  chains.
- verified by the full suite post-migration: failing-case fingerprint
  identical to the pre-migration baseline (the known issues below), all
  chi-square distribution tests green under the new streams.

## vkof: validation message capture

`vkof::validation_message_count()` / `vkof::validation_message(index)` /
`vkof::validation_reset()` (added 2026-07-16, `lib/vkof`) archive every
Vulkan validation-layer message (warning/error severity; debug-printf and
info-severity messages are excluded, same filtering vkof's own stderr log
already applied) since the last `validation_reset()` call. Previously these
only went to stderr, easy to miss in CI output -- now a test can do:

```cpp
vkof::validation_reset();
// ... do the thing under test ...
CHECK(vkof::validation_message_count() == 0u);
```

Smoke-tested in `test-vkof-validation.cpp` on the no-error path (reset
clears the log; a benign dispatch logs nothing). Not yet used to
deliberately provoke and assert on a real validation error -- that's the
natural next step, especially paired with Vulkan's synchronization
validation feature (confirmed enabled cleanly via the `VK_LAYER_VALIDATE_SYNC=1`
environment variable, zero code changes, against this same binary) as a way
to investigate the two open issues below, both of which smell like a
synchronization or lifetime bug.

## Known long-term issues

### FIXED (2026-07-23): unguarded degenerate half vector in coat/metallic/dielectric-specular eval (NaN * 0 = NaN)

**Where:** `openPbrCoatEvaluateF` (`util-material-openpbr-coat.glsl`),
`openPbrFresnelMetallicEvaluateF` (`util-material-openpbr-metallic.glsl`),
and the test-only `openPbrDielectricSpecularEvaluateF`
(`util-dielectric-specular.glsl`, backs `dielectric_specular_evaluate.comp`
/ `_furnace_integrate.comp` / `_uncompensated_evaluate.comp` -- not reachable
from production rendering, which uses the already-guarded copy in
`util-material-openpbr-microfacet.glsl`).

**Root cause:** all three reconstruct a reflection half vector `hLocal =
normalize(wiLocal + woLocal)` with no guard for the `wi == -wo` exact-
retroreflection case, where `wiLocal + woLocal` is the zero vector and
`normalize()` returns NaN. The visibility term (`utilMicrofacetSmithGgx-
VisibilityAniso`) *does* correctly return 0 for this configuration once
`wi`/`wo` disagree in sign, but `NaN * 0 = NaN` in IEEE 754, not 0 -- so the
NaN still poisons the final lobe color. `util-material-openpbr-microfacet.
glsl`'s own `openPbrDielectricSpecularEvaluateF` was already hardened
against exactly this (found 2026-07-17 via the DragonAttenuation/
DispersionTest furnace probes, see git history), but the fix was never
propagated to these three siblings that share the identical pattern.

Found by a dedicated Explore-agent sweep of the remaining unaudited
material shader files, looking for recurrences of this session's two now-
confirmed bug classes (unclamped below-horizon pdfs, missing half-vector
validity checks) -- verified against source before fixing, not taken on
faith.

**Fix:** mirrored the existing guard from the reference function in each
site, adapted per function:
- coat and the test-only dielectric-specular copy also call
  `utilMicrofacetFresnelDielectric` directly (the same eta==1 Inf hazard
  the reference guard's comment documents), so both guards apply: the
  zero-vector check, and a `dotHWi <= 0` back-facing-microfacet check.
  Coat's function has real work left to do after the reflection term
  (coat transmittance/darkening on `baseSubstrate`, which doesn't depend on
  `hLocal`), so its fix zeroes `fCoat` via a safe placeholder `hLocal`
  rather than early-returning the whole function -- an early return there
  would have incorrectly skipped that h-independent work too.
- metallic's `D` term has no Fresnel-Inf hazard of its own (the f82 model
  doesn't divide by an eta-dependent denominator), so only the zero-vector
  guard on `D` is needed; proved `dotHWi > 0` is automatically guaranteed
  whenever `hLocal` is valid and `wiLocal.z`/`woLocal.z` are both positive
  (the only case the visibility term is nonzero for), so no separate
  `dotHWi <= 0` guard is needed there.

**Verified:** full suite still 350/350 test cases passing after the fix
(same as before -- this bug wasn't caught by any existing numeric
assertion, only reachable via the furnace sweep's clearcoat/metallic
assets in a way that a `NAN_CHECK` debug probe would log but not fail on).

### FIXED (2026-07-22): shader-compile bugs silently running tests against stale cached SPIR-V

Found while establishing a baseline before a general bug-hunt pass. Two
independent missing-declaration bugs, both root-caused via
`failed to compile shader ...; will retry on next save` messages in the
raw test output (easy to miss -- vkof falls back to the last successfully
compiled `.spv`, which for a shader that's *never* compiled successfully
can mean silently testing stale/wrong/pre-existing binaries checked into
the repo rather than current source):
- `lib/ponder/shaders/util-shading-frame.glsl` uses the `skTau` macro
  (`utilCosineHemisphereSampleWo`) but never included
  `util-random.glsl`, where it's defined -- it relied on whichever
  `.comp` happened to include `util-random.glsl` first. Any shader that
  included it without independently including `util-random.glsl` failed
  to compile outright: every `shading frame: *` test and two
  `dielectric_specular_*` tests (10 test cases). Fixed by having
  `util-shading-frame.glsl` include its own dependency, matching the
  precedent `util-vdb.glsl`/`util-material-openpbr.glsl` already set for
  this exact situation.
- `unit-test/shaders/vdb_sampling.comp` pulls in `util-random.glsl`'s
  `fnSampleBluenoise` (via `util-vdb.glsl`) but never declared
  `#extension GL_EXT_nonuniform_qualifier` or the `vkofTextures` bindless
  binding. `vdb density: gpu sampling matches cpu nanovdb accessor`
  reported "469 mismatches" -- a real-looking failure number that was
  actually a stale-cached-SPIR-V artifact; fixing the compile error made
  it pass with 0 mismatches. Fixed by adding both declarations.
- Fixing the first bug surfaced three more shaders that transitively gained
  a dependency on `util-random.glsl` through `util-shading-frame.glsl` but
  hadn't declared the extension/binding themselves
  (`shading_frame_xy_crosscheck.comp`, `shading_frame_from_tangent_
  evaluate.comp`, `shading_frame_rotate_evaluate.comp`) -- fixed the same
  way.

**Net effect:** 17 -> 6 failing test cases in the full suite (2 of which,
the kulla-conty ones, are intentionally-expected/documented failures, so
really 17 -> 4 real regressions before the diffuse-pdf fix above, now 3).
**Lesson for future investigation in this file:** always check the raw
test output for `failed to compile shader` before trusting a numeric
mismatch/count as a real finding -- a compile failure here doesn't halt
the test, it silently substitutes old binary output.

### GPU resource leak at shutdown (zeltner LUT + associated views/sampler) -- intentional, not a mystery

**Where:** `test-mixture-lobe-selection.cpp`'s `shared_zeltner_handle()`
(the function's own header comment explains it): a `static
ponder::ZeltnerTables const tables = ponder::zeltner_tables_create();`
function-local static, created lazily once and shared across every test
case in that file, deliberately never paired with a
`ponder::zeltner_tables_destroy()` call.

**Symptom:** a full-suite run's `vkDestroyDevice()` reports Object Tracking
validation warnings for the corresponding undestroyed `VkImage`
(`zeltner-tables.cpp:551`), two `VkImageView`s, a `VkSampler`, and a
`VkDeviceMemory`.

**Status:** this was never a bug to root-cause -- it's a still-open
diagnostic probe a prior session deliberately built and left in place: the
file previously called `zeltner_tables_create()`/`destroy()` per-TEST_CASE
(matching `test-fuzz-eval.cpp`'s convention), and switched to one
shared/leaked handle specifically to test whether the *repeated
create/destroy cycle* (and its bindless-slot churn) was the trigger for the
"diffuse+specular order-dependent chi-square flakiness" below. I mistook
the leftover leak warning for a fresh, unexplained finding on 2026-07-16
before reading the code's own comment -- worth remembering to check for an
explaining comment before writing up something as new. Whether the
create/destroy-churn hypothesis panned out is itself unresolved (see the
flakiness entry).

**To pick back up:** decide whether the create/destroy-churn hypothesis is
confirmed or ruled out (has anyone compared chi-square stability with vs.
without the shared handle across many runs?); if ruled out, restore
per-TEST_CASE `zeltner_tables_create()`/`destroy()` and drop the leak. If
still open, leave as is -- it costs one leaked image for the life of the
test binary, not a correctness problem for anything else.

### FIXED (2026-07-22): "zero-probability lobes are excluded" -- exact-half-domain split

The exact-50%-split clue below was root-caused and fixed: `openPbrGlossyDiffusePdf`
(`lib/ponder/shaders/util-material-openpbr-diffuse.glsl`) computed the cosine-
hemisphere pdf as plain `dotNorWo * IPI`, unclamped -- for any `dotNorWo < 0`
(exactly half of the `mixture_edge_sweep.comp` sweep's `thetaWo` range, which
spans the full `[0, pi]` sphere) this returns a *negative*, finite pdf. The
mixture combination itself was never affected in production (the only
production call site, `openPbrEvaluatePdfSelected`, only reaches this
function after an early-return that guarantees `dotNorWo > 0`), but the
"zero-probability lobes are excluded" test also evaluates+outputs the raw,
unguarded primitive for whichever lobe it's excluding, and for the excluded-
diffuse case that raw value went negative across exactly the below-horizon
half of the 65536-sample sweep -- matching the documented `otherNegative ==
32768` symptom exactly. Two other test shaders (`sampling_expected.comp`,
`fuzz_pdf_integrate.comp`) had already independently worked around the same
missing clamp externally, which is what tipped this off. Fixed by clamping
inside the primitive itself: `return max(dotNorWo, 0.0f) * IPI;` -- matches
every caller's actual expectation and makes the external workarounds
redundant (left in place, harmless). Verified: the test now passes clean.

### FIXED (2026-07-23): severe transmission+specular chi-square failure -- missing half-vector sidedness check

**Where:** `openPbrTransmissionPdf` and `openPbrTransmissionEvaluateF`
(`lib/ponder/shaders/util-material-openpbr-transmission.glsl`).

**Root cause:** neither function checked that `wi` and `wo` actually land on
*opposite* sides of the reconstructed generalized half vector `h_t =
-(eta_i wi + eta_t wo)` -- a physical requirement for a valid refraction
configuration that Walter et al. 2007's own derivation depends on (this
file's own denom comment already assumed the cancellation this sidedness
guarantees, without the code ever verifying it held) and that reference
implementations (pbrt's `MicrofacetTransmission`, Blender Cycles'
`bsdf_microfacet.h`, both already cited elsewhere in this file) gate on
explicitly: `dot(wi,h) * dot(wo,h) <= 0` required, zero otherwise. Without
it, an algebraically-reconstructed `h` from a same-side `(wi, wo)` pair
still produced a nonzero D/G1/F-based density, even though no real
VNDF-sample-then-`refract()` path can ever land there (`refract()` always
sends the transmitted ray to the *opposite* side of the sampled half
vector).

**How it was found:** the prior session's two-thread diagnostic plan (see
git history for the pre-fix version of this section) settled thread 1 --
a 256x256 fine quadrature of the exact bin-20 outlier bin agreed with the
coarse 32x32 `mixture_expected.comp` value almost exactly (126.464 vs
126.4), ruling out a quadrature-resolution artifact and confirming the
analytic pdf robustly, smoothly claims density there. Probing the actual
half-vector reconstruction at that same point found `dot(wi,h)=0.996` and
`dot(wo,h)=0.998` -- both positive, i.e. `wi`/`wo` on the *same* side of
`h`, the exact non-physical configuration the missing check should have
caught.

**Fix:** added `if (dot(wi, htOriented) * dot(wo, htOriented) > 0.0f)
{ return 0/vec3(0); }` to both functions, right after `htOriented` is
computed. Verified: all 19 `mixture:*`/`DEBUG:*` test cases in
`test-mixture-lobe-selection.cpp` now pass (4609/4609 assertions) -- this
includes not only the transmission+specular failures documented below, but
also the previously-separate, smaller-magnitude "below-horizon leak-branch
residual" in the kitchen-sink config (its own section below is folded into
this one; same missing check, same fix). Full suite: 350/350 test cases
pass (the only 2 remaining assertion failures are the pre-existing,
intentionally `doctest::should_fail`-marked kulla-conty tests documented
elsewhere in this file, not regressions).

**Why this was safe to fix (unlike two prior sessions' hesitation):** the
sidedness requirement is standard, independently-citable physics (not a
self-derived guess) -- both reference implementations already cited
elsewhere in this same file already enforce it, and the fine-quadrature
diagnostic gave a decisive, mechanical (not speculative) signal before any
production code changed.

### diffuse+specular order-dependent chi-square flakiness

**Where:** `test-mixture-lobe-selection.cpp`, the `diffuse+specular`
pairwise chi-square test.

**Symptom:** passes when run standalone; occasionally fails when run
together with certain other test cases selected via `--test-case`, even
when it finishes and reads back its result before those other test cases'
bodies execute — ruling out a simple runtime-order race. Root cause never
found: a shared long-lived Zeltner LUT handle, a `debugPrintfEXT` probe of
the actual push-constant values received, and a trivial-shader
(`fill_u32.comp`)-based harness reproducer (kept at
`test-dispatch-readback-race.cpp`) all failed to explain it. Set aside once
a much clearer, unrelated bug (`openPbrTransmissionTirReflectPdf`
double-counting, now fixed) took priority.

**Status:** open, unexplained. `test-dispatch-readback-race.cpp` is a
ready-made starting point if picked back up.
