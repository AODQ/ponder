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

### severe transmission+specular chi-square failure, distinct from the documented flakiness

**Where:** `test-mixture-lobe-selection.cpp`, isolated by running only
`mixture:*`/`DEBUG:*` (i.e. this file alone, no other test files, ruling
out any cross-file ordering effect).

**Symptom:** 6 of 18 test cases fail reproducibly running this file alone
(37 assertions): the three "aniso specular seam bisection, +transmission*"
variants, "sampler vs pdf chi-square, transmission+specular", "... kitchen
sink", and "zero-probability lobes are excluded". This is **not** the
already-documented "diffuse+specular" flakiness below -- that specific test
case passes in this same run. Two things stand out as more severe than
anything previously written up here:
- the transmission+specular chi-square's `cs.pValue`/`csBelow.pValue` are
  exactly `0` with `cs.statistic` at ~7-9x its `dof` (e.g. statistic
  16902.7 vs dof 2347) -- a decisive blowup, not the "small residual" the
  below-horizon entry below describes.
- "zero-probability lobes are excluded" fails with `poison.otherNegative
  == 32768` out of 65536 swept samples -- **exactly half** the domain,
  which smells like a systematic sign/half-space bug (below-horizon vs
  above, or a phi-range split) rather than numerical noise.

**Status:** found/isolated 2026-07-16, not root-caused -- this is
downstream of the same "digging into leftover leak infrastructure" thread
that turned up the intentional-leak finding above, and reporting it here
rather than continuing into `util-material-openpbr.glsl` itself, per how
deeper investigations get consulted on in this repo. Likely related to (or
a more severe instance of) the below-horizon residual entry below, given
both involve transmission's below-horizon leak term, but not confirmed --
the exact-half-domain split in the exclusion-test failure doesn't obviously
match "small residual concentrated in high-theta bins."

**To pick back up:** the exact-50%-split clue is the strongest lead --
find what parameter in the 65536-sample sweep partitions the domain
exactly in half (phi in `[0, pi)` vs `[pi, 2*pi)`? `dotNorWo` sign?) and
check what `openPbrEvaluatePdfSelected`/the transmission leak term does
differently on each side.

### Below-horizon leak-branch residual in the multi-lobe mixture pdf

**Where:** `openPbrEvaluatePdfSelected`'s `dotNorWo <= 0` branch
(`lib/ponder/shaders/util-material-openpbr.glsl`), specifically when coat,
anisotropic specular, and transmission are all simultaneously active (the
"kitchen sink" config in `test-mixture-lobe-selection.cpp`).

**Symptom:** small chi-square residual (`csBelow.pValue` fails) concentrated
in below-horizon bins (`thetaBin` past 20 of 40), visible as a blue band in
`mixture_kitchen_sink_chi2_ct*.png` — the expected (quadrature) side
predicts ~2-8% more density per bin than the observed (sampler) side
produces there. Small in magnitude, not the severe blowup a prior bug in
this same milestone had (see `openPbrTransmissionTirReflectPdf`'s fix
history in git log / the mixture-lobe-selection milestone's own
investigation).

**Status:** found 2026-07-16, not yet root-caused. Distinct from the
already-fixed above-horizon TIR-reflect double-counting bug — this one
lives in the below-horizon combination of `openPbrTransmissionPdf` +
specular-leak + coat-leak terms, which was never independently re-verified
with all three simultaneously nonzero after that fix landed. Deprioritized
since it's small and the milestone's higher-value findings (the TIR fix,
port fidelity, selection-weight crosscheck) are settled.

**To pick back up:** bisect which pairing of {coat-leak, specular-leak,
transmission} below horizon reproduces it in isolation (same technique
used for the TIR bug: pairwise isolation test cases, then a sub-grid scan
of `mixture_expected.comp`'s quadrature to pinpoint the exact `(wi, wo)`
where expected exceeds observed).

### NaN in transmissionTirPdf under bluenoise sampling (DragonAttenuation)

**Where:** `openPbrTransmissionTirReflectPdf`
(`lib/ponder/shaders/util-material-openpbr-transmission.glsl`), reached via
`openPbrEvaluatePdfSelected`'s above-horizon branch.

**Symptom:** rendering `assets/Models/DragonAttenuation` through the real
`viewer` binary (`unit-test/scripts/render-furnace-models.sh`, furnace mode,
320x240 @ 64spp, bluenoise-seeded) hits `px(145,97)` with
`transmissionTirPdf` = NaN, cascading through `pdfAboveHorizon` ->
`openPbrSampleWo.pdf` -> `openPbrEvaluateF.fSpecular` (all `NaN`, caught by
the standing `util-nan-probe.glsl` checks). `test-furnace-model-render.cpp`'s
own DragonAttenuation case (64 deterministic per-pixel samples, no
bluenoise) never hits this exact `(wi, wo)` config, which is why it wasn't
caught by that numeric test.

**Status:** found 2026-07-17, immediately after landing the
isotropic->anisotropic-distribution fix to this same function's peak-D
mismatch (see `project_dragon_tir_firefly_rootcause` memory / git log for
that fix). Confirmed **pre-existing, not caused by that fix**: bisected via
`git stash` on just this file, reran the identical viewer command against
the old (isotropic-only) code, same NaN at the same pixel. Most likely
`h = normalize(wi + wo)` going degenerate (`wi` close to `-wo`) somewhere
downstream of the `refract()`/TIR check, or a similar 0/0 in the D/G1 chain
that a narrower, deterministic sample set doesn't happen to walk into.

**To pick back up:** reproduce standalone (a unit-test shader dispatch at
exactly this pixel's `(wi, wo, mat)`, sidestepping the need to re-render
through the full viewer each time), then trace which term inside
`openPbrTransmissionTirReflectPdf` first goes non-finite -- same
pixel-gated-probe technique as the firefly investigation
(`gNanProbeCoord.x==145 && y==97`, disable every other active probe first
to avoid budget starvation).

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
