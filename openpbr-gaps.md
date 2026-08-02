# OpenPBR conformance gaps

Audit of `lib/ponder/shaders/util-material-openpbr*.glsl` + `app/viewer/shaders/util-raytrace.glsl`
against the OpenPBR spec and its normative MaterialX reference (`reference/open_pbr_surface.mtlx`).

Refs below: `spec` = academysoftwarefoundation.github.io/OpenPBR (equation labels quoted),
`mtlx` = the reference implementation with node names + line numbers.

All 40 spec parameters exist in `OpenPbrMaterial` and are referenced somewhere. Every gap
here is a coupling/direction/wiring issue, not a missing parameter.

---

## A. Layering blocking factor uses the wrong direction  — FIXED

**Spec.** §Slabs fixes the convention: "ω_o in the direction of the outgoing light and ω_i
opposite to the direction of the incident light", with
`E(ω_o) ≡ ∫ f(ω_i,ω_o) dω_i^⊥` (eq. `directional-reflectance-definition`).
Eq. `non-reciprocal-albedo-scaling`: `f_layer = f_coat + (1 − E_coat(ω_o))·f_sub`.
The factor is at the **camera-side** direction — that is the whole point, so it pulls out of
the light integral and `E_layer(ω_o) ≤ 1` holds.

**Ponder's convention.** Camera-side is `wi`: `itWi = -bsdfWo` (`util-raytrace.glsl:686`),
and `openPbrEvaluateF(..., itWi, envDir, ...)` (`:326`) integrates over the *second* argument.
So ponder `wi` == spec `ω_o`.

**Current state — three layers, three conventions:**

| layer | evaluated at | verdict |
|---|---|---|
| fuzz (`fuzz.glsl:119`) | `dot(nor, wi)` | correct |
| lobe selection (`util-material-openpbr.glsl:254`) | `dotNorWi` | correct |
| base specular (`util-material-openpbr.glsl:184`) | `dot(nor, wo)` | **wrong** |
| thin-walled sheet `essThin` (`util-material-openpbr.glsl:116`) | `abs(dot(nor, wo))` | **wrong** |
| coat (`coat.glsl:169-176`) | both `μi` and `μo` | correct term + an extra one (see D) |

**Why the tests didn't catch it.** `unit-test/shaders/glossy_diffuse_layer_furnace_integrate.comp:70-88`
fixes `wo`, calls it the view direction, and its own comment says: *"fixing wi instead and
integrating over wo (the more obvious-looking choice) does NOT give a clean identity: Rs(wo)
would then vary across the integration domain instead of factoring out."* That is exactly what
the renderer does. The test validates the algebra in the convention where it works; the
renderer uses it in the convention where it doesn't.

**Symptom.** A diffuse base lit at grazing incidence is darkened by the *light's* Fresnel
instead of a constant; terminator darkening; a coat/base white furnace test in the renderer's
convention fails.

**Fix — applied.** `util-material-openpbr.glsl:184` and `:116` now read `dot(nor, wi)`.
Still to do: flip `glossy_diffuse_layer_furnace_integrate.comp` to fix `wi` and integrate `wo`,
so the test runs in the renderer's convention instead of the one that hid this.

---

## B. Reflection and transmission use different IORs  — FIXED

**Spec.** §Specular: "The Fresnel transmission factor and refraction into and out of the base
dielectric should also be consistent with the IOR ratio η′_s."
**mtlx.** One `modulated_eta_s` node (`:367`) feeds *both* `dielectric_reflection` (`:396`) and
`dielectric_transmission` (`:379`).

**Current.** Reflection uses `openPbrEffectiveIor(specularIor, specularWeight)`; transmission
gets raw `material.specularIor` via `dispersedIor` (`util-raytrace.glsl:303` →
`transmission.glsl:104-105`). At `specular_weight = 0` the reflection correctly vanishes but
refraction still bends at 1.5. Same root cause in dispersion: spec says `specular_ior`
"including any modulation via specular_weight" defines n(λ_d), but
`transmission.glsl:44` (`openPbrDispersionIorRgb`) uses the raw value.

**Fix — applied.** New `openPbrTransmissionIor(mat)` in `transmission.glsl` returns
`openPbrEffectiveIor(specularIor, specularWeight)`. `openPbrDispersionIorRgb` uses it as n_d in
both branches; `util-raytrace.glsl` uses it for the non-dispersive `dispersedIor` fallback and
for the subsurface exit interface (the four `exitMaterial.specularIor` call sites now share the
existing `etaI` local). `transmission_dispersion_evaluate.comp` had to set
`specularWeight = 1.0f` — it left the field uninitialized and the function now reads it; 1.0 is
the identity, so its golden values are unchanged.

---

## C. Coat does not modify the base specular IOR  — FIXED

**Spec.** eq. `specular_ior_ratio`: `η_s = lerp(n_b/n_a, n_b/n_c, C)`, plus the Kutz TIR fix
(eq. `specular_ior_ratio_with_tir_fix`) swapping to `n_c/n_b` when `n_c > n_b`.
**mtlx `:305-321`** implements it exactly:

```
specular_to_coat_ior_ratio = specular_ior / coat_ior
tir_fix = (ratio > 1) ? ratio : coat_ior / specular_ior
eta_s   = mix(specular_ior, tir_fix, coat_weight)
```

**Current.** `openPbrEffectiveIor` (`microfacet.glsl:947`) has no coat term; no call site passes
one (`util-material-openpbr.glsl:109,139,257`, `ibl.glsl:263`).

**Magnitude.** At defaults (`coat_ior` 1.6 > `specular_ior` 1.5) with full coat, η_s goes
1.5 → 1.0667, so base F0 goes 0.04 → 0.00107 — ~37× less base highlight under a coat.

**Fix — applied.** `microfacet.glsl` gains `openPbrCoatAffectedIor(specularIor, coatIor,
coatWeight)` (TIR-fixed lerp, early-out at `coatWeight <= 0`) and
`openPbrSpecularEffectiveIor(mat)` which composes it with the specular-weight modulation.
All five call sites now go through it: `util-material-openpbr.glsl:109,139,257`,
`ibl.glsl:263`, and `openPbrTransmissionIor` — the reference feeds the same `modulated_eta_s`
to reflection and transmission, so B's helper folds in here too.

**Minor, same function — applied.** `openPbrEffectiveIor` now carries the `sgn(η_s − 1)` of
eq. `modulated_ior` (mtlx `:352`). Identity for η_s > 1.

**Not changed — follow-up.** `coat.glsl:193`'s `baseFresnelWo` (the eq. 69 base-roughness
estimate) still evaluates F_s at the raw `specularIor`. Per eq. `unmodulated_fresnel` that
should also be the coat-affected η_s. Left alone to keep this change scoped; it only feeds the
darkening heuristic.

**Expect golden-value shifts** in any test that drives `coatWeight > 0` through the full stack
(`mixture_*`, `cornell_openpbr_render`, `dielectric_scene_render`, `openpbr_ibl_evaluate`,
`subsurface_entry_selection_fraction`). That is the fix working, not a regression — rebaseline
after reviewing. Tests that pin `coatWeight = 0` or pass `etaRel` directly are unaffected.

---

## D. Emission is not attenuated by the coat  — MEDIUM

**Spec.** §Emission places emission below the coat *specifically* so it is tinted;
lobe reduction gives `L_e = lerp(1, T_coat, C)·E`.
**mtlx `:583-608`** is more specific:

```
emission_edf = mix(uncoated, coated, coat_weight)
coated       = generalized_schlick_edf(color0 = 1 − F0_coat, color90 = 0, exp 5)
                 over (uncoated × coat_color)
```

i.e. coat absorption tint *plus* the coat interface's directional transmittance on the way out.

**Current.** `util-raytrace.glsl:255` adds `emissionColor * emissionLuminance` raw at every
vertex. The layer diagram in `util-material-openpbr.glsl:20-29` already draws the emission
arrow passing through the coat.

**Fix.** Entirely local to the emission site in `util-raytrace.glsl` — it needs `coat_ior`
(for `F0`), `coat_color` and `coat_weight`, all already on the material, and does **not** reuse
the coat lobe's directional albedo. So it touches no shared helper and nothing else depends on
it: a leaf change, orderable anywhere.

The one direction it does need is the view cosine, since `generalized_schlick_edf` is
directional. At the emission site that is `dot(itFrame.nor, itWi)` — the camera-side direction,
same convention finding A settled.

**Not** a gap: fuzz attenuation of emission — the reference omits it and the spec defines fuzz
as purely scattering / non-absorbing.

---

## E. Negative absorption clamped per-channel instead of shifted achromatically  — FIXED

**Spec.** §Translucent base: if any component of `μa = μt − μs` is negative, shift by
`μa ← μa − min(μa)` (gray, all channels), then `μt = μa + μs`.
**mtlx `:258-282`**: per-channel min, subtract from all three, gated on `min < 0`.

**Current.** `util-raytrace.glsl:1133`: `max(σt − σs, 0)` per channel — shifts hue and lowers
total extinction where the spec preserves both.

**Fix — applied.** `util-raytrace.glsl` now computes the raw `μt − μs`, takes the channel min,
and subtracts it from all three only when negative. `openPbrSubsurfaceWalkStep` derives `μt`
from `(μa, μs)`, so the extinction follows automatically.

---

## F. Tangent maps loaded then discarded  — FIXED

**Spec.** §Normal maps: `geometry_tangent` / `geometry_coat_tangent` define anisotropy direction.
**mtlx** wires `geometry_tangent` into every base BSDF (`:382,399,407,433,441`) and
`geometry_coat_tangent` into `coat_bsdf` (`:545`).

**Current.** `openPbrLoadMaterialDeriv` returns them, then `util-raytrace.glsl:651-662` drops
`tangentNormal` and `modelClearcoatNormal`. Anisotropy direction comes only from the mesh
tangent plus the glTF `specularRoughnessAnisotropyRotation` scalar; the coat has no rotation
control at all, so `coat_roughness_anisotropy` is always mesh-tangent aligned.
`geometryCoatTangentTexture` has zero uses repo-wide.

**Fix — applied.** `MOR_MATERIAL_ALL_PARAMS_NORMAL` in `mor-shared.h` gained a third arg: the
value each map decodes to when unbound. Normals keep +z; tangents now get **+x**, so
`tbn * identity` reproduces the mesh tangent and the unbound path needs no branch.
`LOAD_COMPONENT_NORMAL` (both copies in `openpbr-load.glsl`) takes and uses it. Both consumers
now build their frames from `tbn * geometryTangent.xyz` / `tbn * geometryCoatTangent.xyz`:
`util-raytrace.glsl` (locals renamed off the misleading `tangentNormal` /
`modelClearcoatNormal`) and `scene-visibility-resolve.comp`. `specularRoughnessAnisotropyRotation`
still composes on top as a scalar offset.

**One delta on the default path.** With no tangent map bound the frame tangent is now
`tbn * (1,0,0)` — orthogonalized against the interpolated normal, then again against the mapped
normal by `shadingFrameFromTangent` — where it used to be the raw `tangent.xyz` orthogonalized
once. Identical unless a normal map pulls the shading normal far off the interpolated one.
Degenerate mesh tangents now inherit `utilCalculateTbnBasis`'s Frisvad fallback, which is the
same basis the normal mapping already uses.

---

## G. Coat layering is two-sided; spec's is one-sided  — FIXED

Eq. `non-reciprocal-albedo-scaling` uses a single `(1 − E_coat(ω_o))` and is *named*
non-reciprocal, so one-sided is deliberate. `coat.glsl:169-176` uses
`(1 − E(μi))(1 − E(μo))`. It contains the correct ω_o term, so this is an extra factor rather
than a wrong one — unlike A.

**Fix — applied.** `coat.glsl` now blocks the base with the single `(1 − E_coat(ω_o))`,
evaluated at `wiLocal.z` — the camera-side direction in this renderer's convention, which is
what lets it factor out of the light integral. `coatAlbedoWo` is gone. `ibl.glsl`'s
`(1 − coatAlbedo)²` collapses to one factor to match.

**Expect the coated base to brighten**, most at grazing angles where the dropped second factor
was largest. `coat_ibl_furnace_integrate` should move *toward* 1, not away — if it doesn't,
that is a signal worth chasing.

---

## H. Thin film missing from the transmission lobe  — LOW (reference also omits)

Spec prose is explicit ("the thin film should also generate color fringes in the transmission
lobe... soap bubbles"). But mtlx applies thinfilm only to `dielectric_reflection_tf` (`:402`)
and `metal_bsdf_tf` (`:435`); `dielectric_transmission` (`:377`) gets none.
`util-material-openpbr-transmission.glsl` has zero thinfilm references.

Real spec gap, but implementing it puts you ahead of the reference — a MaterialX crosscheck
would then *disagree*. Gate on whether crosscheck parity or spec fidelity wins.

### Research: what applying it actually costs

**The risk that would have killed it isn't there.** Neither `openPbrTransmissionPdf` nor
`openPbrTransmissionTirReflectPdf` uses a Fresnel term — they are pure VNDF/Jacobian and
D·G1 respectively — and `openPbrSampleWo`'s transmission branch picks refract-vs-TIR from
`refract()` returning zero length, not a Fresnel coin flip. So making F chromatic in the eval
desyncs no pdf and no MIS weight. The edit inside `transmission.glsl` is two lines: the `F` at
`:130` (thin-walled / index-matched branch) and `:178` (rough branch) become `vec3`, and
`(1 - F)` a per-channel multiply. The BTDF already returns `vec3`.

**Note which branch matters most.** Soap bubbles — the spec's own motivating example — are
thin-walled, so `:130` is the branch that carries the use case, not the rough refraction one.

Three things need deciding before it lands:

1. **T = 1 − R is only valid if the RGB result is normalized.** The spec licenses the identity
   ("the thin film is non-absorbing... it only redistributes the probabilities of reflection
   and transmission"). But `openPbrThinfilmFresnel` returns `xyzToRgb * I`, an XYZ-integrated
   reflectance, and `1 − R_rgb` equals `∫(1−R(λ))s(λ)dλ` only if a flat `R(λ)=1` maps exactly
   to `(1,1,1)`. Belcour normalizes for this (`S0` at opd 0, the `/1.0685e-7` divisor), but it
   is a Gaussian *fit* — verify numerically before relying on it. `thinfilm_evaluate` /
   `thinfilm_crosscheck` already have the harness: check that an index-matched film reproduces
   the plain dielectric Fresnel, and that a fully-reflecting configuration returns ~1.

2. **The film's outer medium is hardcoded to air.** `openPbrThinfilmFresnel` passes
   `etaI = 1.0f` into the first interface. Exiting from inside the medium, the film's outer
   side is the dense medium, not air — so the exit-side film is unmodelled. The reflection lobe
   already lives with this; for transmission it is more visible because you cross the
   interface. Either accept it (consistent with reflection) or generalize the function to take
   an incident ior.

3. **The subsurface exit's Fresnel cancellation breaks.** `util-raytrace.glsl`'s exit branch
   russian-roulettes on a *scalar* `fresnelExit`, then divides the eval by `(1 - fresnelExit)`
   to cancel the eval's own `(1 - F)`. If the eval's F becomes chromatic, a scalar no longer
   cancels a vec3 and that path is biased per channel. Affects `subsurface_weight > 0` combined
   with `thin_film_weight > 0`. Needs either a chromatic `fresnelExit` (branch on a hero or
   luminance channel, divide by the matching per-channel value) or an explicit carve-out.

Cost: the film evaluation is a 4-order Airy sum over two polarizations plus the XYZ
sensitivity, and transmissive surfaces would now pay it twice per hit (reflection + BTDF).
Worth hoisting to one call per shading point if this lands.

### How Cycles does it

Blender/Cycles ships thin film on the transmission side, so unlike MaterialX there is a working
implementation to compare against (`intern/cycles/kernel/closure/bsdf_microfacet.h`,
`bsdf_util.h`). It settles two of the three questions above.

**Q1 — `T = 1 − R` is what a production renderer does.** `generalized_schlick_fresnel` ends with

```c
*r_reflectance   = F * fresnel->reflection_tint;
*r_transmittance = (one_spectrum() - F) * fresnel->transmission_tint;
```

where `F` is the *iridescent* per-channel reflectance. So the identity is used directly, no
separate Airy transmittance.

Caveat that keeps Q1 open for us: Cycles evaluates iridescence **per channel spectrally**
(`FOREACH_SPECTRUM_CHANNEL` → `fresnel_iridescence_channel(kg, i, ...)`), so `1 − R` is exact
per channel. Ponder integrates against the Belcour XYZ Gaussian fit and then applies
`xyzToRgb` and a `clamp(0,1)`. `1 − R_rgb` is therefore only as good as that fit's
normalization. The numeric check stands — it is a ponder-architecture question, not a
model question.

**Q2 — solved, and smaller than expected.** Cycles does not generalize the film function to
take an incident IOR. It rescales, with the derivation in-source
(`adjust_thin_film_ior_at_backface`): the backface stack `bulk | film | 1` is equivalent to
`1 | film/bulk | 1/bulk`, so it divides `film_ior` by `bulk_ior` and evaluates against a base
IOR of `1/bulk_ior`. That maps onto `openPbrThinfilmFresnel`'s hardcoded `etaI = 1.0f` with no
signature change — pass `thinFilmIor / bulkIor` and `etaBase = 1/bulkIor` on the exit side.

**Q3 — still ours.** Cycles returns reflectance and transmittance as a pair and samples the
branch from them, so the scalar-vs-chromatic cancellation in our subsurface exit has no
analogue there. That integration point remains ponder-specific.

**Bonus: the thin-walled path should be more than `1 − F`.** `bsdf_thin_glass_fresnel`
evaluates the film at *both* interfaces — front `(r1,t1)`, and back `(r2,t2)` using the
backface-adjusted film IOR — then sums the internal reflections in the shell:

```c
c = transmission_tint^(-1/cos_theta_t);          // beer-lambert at oblique angle
*r_transmittance = c*t1*t2 / (1 - sqr(r2*c));    // t' = ct1t2 + ct1(r2c)^2t2 + ...
*r_reflectance   = r1 + *r_transmittance*r2*c;   // r' = r1 + ct1r2ct2 + ...
```

This is the soap-bubble path, and it also modifies the **reflection** lobe — precisely the
effect OpenPBR §Thin-walled mode flags and then declines to model ("the reflection lobe from
the dielectric will also technically be modified due to the internal bounces in the sheet").
Our `:130` branch returns `tint * (1 - F) / dotNorWo`, i.e. one interface, no series. If the
goal is soap bubbles specifically, this series is the thing that produces the look, not the
`1 − F` substitution.

**Unrelated difference worth knowing.** Cycles combines the F0 tint with iridescence via a
bespoke scaling (`mix(1, f0/F0_real, s)` with `s` from how close `F` is to `F0_real`) because
"F may be below F0_real". Ponder lerps `mix(dielectricFresnel, thinfilmFresnel, thinFilmWeight)`,
which follows KHR_materials_iridescence's factor semantics instead. Different, not wrong.

---

## Investigated and withdrawn

| claim | why it's not a gap |
|---|---|
| fuzz should use a coat/base blended normal | mtlx `sheen_bsdf` uses `geometry_normal`, same as ponder. Prose-only. |
| thin film should know the coat is above it | mtlx thin-film has no outer-medium input at all. Spec calls the 8-config treatment "implementation-dependent". |
| `specular_color` should not tint reflections from below | mtlx tints the `scatter_mode="R"` BSDF with no side distinction (`:397`). Prose-only. |

---

## Where ponder is ahead of the reference

Do **not** "fix" these toward MaterialX — expect crosscheck deltas that are the reference being cruder.

- **Coat view-dependent absorption.** `coat.glsl:146-168` implements spec eq. 71/72;
  mtlx `coat_attenuation` (`:526`) is a flat `mix(1, coat_color, coat_weight)`.
- **Coat darkening `E_base`.** mtlx (`:475-489`) uses `base_color × specular_weight` for the
  metal albedo and omits `transmission_color`; `coat.glsl:211-231` uses `baseWeight × baseColor`
  and includes the transmission slab — closer to the spec's "normal-incidence albedos of the
  individual slabs, blended according to their mix weights".

---

## Test coverage holes

- **No PT-path coat furnace or coat crosscheck.** Only `coat_ibl_*` (raster) and
  `coat_roughened_roughness_evaluate` exist. Every other lobe has one. This is exactly where
  A and C live.
- **Furnace test convention.** Existing layer furnace tests fix `wo`; the renderer fixes `wi`.
  At least one test should run in the renderer's convention.
- Spec §White furnace testing also lists, untested here: translucent base with white
  `transmission_scatter` at depth > 0; mixed-weight configurations; invariance under
  normal-mapped base/coat and under `geometry_opacity`.
- No emission test; no coat-darkening crosscheck.

---

## Suggested order

A, B, C, E, F and G are applied (shaders + C++ compile clean; suite not yet re-run).

Remaining **D** and **H** are independent of each other and of everything already landed —
there is no dependency ordering left, only cost. D is a leaf (one site, no shared helper);
H is two changes with an open numeric question (see its research section).

The actual blocker is neither: **run the suite.** Six findings have landed and several
deliberately move golden values (C and G especially). Stacking more onto an unverified pile is
the only real sequencing risk here.

Also outstanding from A: flip `glossy_diffuse_layer_furnace_integrate.comp` into the renderer's
direction convention, and add the missing PT-path coat furnace test — the thing that would have
caught A and G in the first place.
