# ponder

A Vulkan path tracer built around a full OpenPBR material model. Every image
below is produced by the unit-test suite (`unit-test/output/`) — scene renders
are validation artifacts, heatmaps are the visual output of the statistical
and numerical checks each BSDF lobe went through.

# TODO
- vdb:
 - HDDA to skip empty space in vdb blobs
 - fp16 vdb grids
 - vdb grid placement transform
 - multiple vdb blobs
 - per-blob albedo, sigma, droplet size
 - additional vdb grid imports
- surface path-tracer:
 - openpbr SSS lobe
 - scene light NEE (currently only environment light NEE)
- rasterizer:
 - port old gpu occlusion culling
 - port old ddgi probes, maybe update to ReSTIR?
 - maybe implement LOD selection?
- overall:
 - need a better tonemapper/hdr system
 - need bloom (maybe?)
 - need denoiser

## Cornell box — OpenPBR material sweep

Full ported OpenPBR chain (lobe selection → sample → evaluate/pdf), two
rotated boxes on real TLAS instance transforms.

![reference Lambertian path tracer, hand-rolled walls](unit-test/output/cornell-box.png)
*Reference: hand-rolled Lambertian path tracer.*

![same walls as OpenPBR diffuse surfaces](unit-test/output/cornell-openpbr-classic-diffuse.png)
*Agreement check: the same walls expressed as OpenPBR surfaces — color bleed and falloff match the reference.*

![near-perfect silver mirror boxes](unit-test/output/cornell-openpbr-mirror-silver.png)
*Near-smooth silver mirror.*

![mirror walls](unit-test/output/cornell-openpbr-mirror-walls.png)
*Mirror walls — multi-bounce specular interreflection.*

![brushed gold](unit-test/output/cornell-openpbr-brushed-gold.png)
*Brushed gold — anisotropic metal.*

![anisotropy direction comparison](unit-test/output/cornell-openpbr-brushed-aniso.png)
*Anisotropic roughness with differing tangent orientation per box.*

![copper vs steel](unit-test/output/cornell-openpbr-copper-vs-steel.png)
*Copper vs. steel — F82 metallic base-color tint.*

![cherry vs navy paint](unit-test/output/cornell-openpbr-cherry-vs-navy.png)
*Colored glossy-diffuse (paint-like) surfaces.*

![glossy plastic](unit-test/output/cornell-openpbr-glossy-plastic.png)
*Smooth plastic — dielectric specular over diffuse base.*

![rough plastic](unit-test/output/cornell-openpbr-rough-plastic.png)
*Rough plastic.*

![high vs low ior](unit-test/output/cornell-openpbr-high-vs-low-ior.png)
*Specular IOR sweep — high vs. low index side by side.*

![EON sigma sweep](unit-test/output/cornell-openpbr-eon-sigma-sweep.png)
*EON rough-diffuse roughness (σ) sweep.*

## Glass & instancing scenes

Medium-tracking path tracer: nested dielectrics, hollow shells, TLAS at scale.

![nested thin shell dense core](unit-test/output/cornell-glass-nested-thin-shell-dense-core.png)
*Nested dielectrics: thin shell (η 1.05) around a dense core (η 2.40).*

![nested dense shell thin core](unit-test/output/cornell-glass-nested-dense-shell-thin-core.png)
*Dense shell (η 2.40) around a barely-there core (η 1.05).*

![nested glass shell water core](unit-test/output/cornell-glass-nested-glass-shell-water-core.png)
*Glass shell (η 1.50) around a water core (η 1.33).*

![nested matched ior tinted core](unit-test/output/cornell-glass-nested-matched-ior-tinted-core.png)
*Matched IOR, tinted core — confirms color and bend angle vary independently.*

![checkerboard ior 1.05](unit-test/output/cornell-glass-checker-ior-thin-1.05.png)
*Hollow shell over a checkerboard, η 1.05 — barely bends.*

![checkerboard ior 1.33](unit-test/output/cornell-glass-checker-ior-water-1.33.png)
*Same shell at water's index, η 1.33.*

![checkerboard ior 1.5](unit-test/output/cornell-glass-checker-ior-glass-1.5.png)
*Ordinary glass, η 1.50.*

![checkerboard ior 2.4](unit-test/output/cornell-glass-checker-ior-diamond-2.4.png)
*Diamond, η 2.40 — strongest refraction and internal reflection.*

![2500 bouncing spheres](unit-test/output/cornell-bouncing-spheres.png)
*2,500 instances (4 size tiers × 6 material kinds) — TLAS instancing at scale.*

## Furnace tests on glTF models

Real textured models path-traced under a uniform environment with no lights —
every pixel should converge to the environment intensity; any firefly or NaN
is a library bug.

![furnace avocado](unit-test/output/furnace-avocado.png)
*Avocado — first scene-level furnace/NaN test.*

![furnace dispersion test](unit-test/output/furnace-dispersion-test.png)
*DispersionTest — transmission + dispersion under furnace conditions.*

![furnace dragon attenuation](unit-test/output/furnace-dragon-attenuation.png)
*DragonAttenuation — the scene that exposed (and now guards) the TIR-fallback pdf/eval anisotropy mismatch.*

## Ray casting

![corner tunneling](unit-test/output/raycast-corner-tunneling.png)
*Corner-tunneling regression — rays with tMin=0 escaping through shared BLAS edges.*

![shadow acne](unit-test/output/raycast-shadow-acne.png)
*Shadow-acne / self-intersection check.*

## Microfacet foundations (GGX, Smith, Fresnel)

![ggx isotropic heatmap](unit-test/output/ggx_isotropic_heatmap.png)
*GGX distribution over angle × roughness.*

![smith g1](unit-test/output/smith_g1_heatmap.png)
*Smith G1 masking term.*

![smith visibility](unit-test/output/smith_visibility_heatmap.png)
*Smith height-correlated visibility (G2).*

![fresnel dielectric](unit-test/output/fresnel_dielectric_heatmap.png)
*Dielectric Fresnel over incidence × IOR.*

![fresnel metallic untinted](unit-test/output/fresnel_metallic_heatmap_untinted.png)
*F82 metallic Fresnel, untinted.*

![fresnel metallic gold tinted](unit-test/output/fresnel_metallic_heatmap_gold_tinted.png)
*F82 metallic Fresnel with gold edge tint.*

![directional albedo](unit-test/output/directional_albedo_heatmap.png)
*Directional albedo over incidence × roughness (energy-compensation input).*

![kulla conty table](unit-test/output/kulla_conty_table_heatmap.png)
*Kulla–Conty energy-compensation lookup table.*

![kulla conty analytic](unit-test/output/kulla_conty_analytic_heatmap.png)
*Analytic energy-compensation reference.*

![kulla conty diff](unit-test/output/kulla_conty_vs_analytic_diff.png)
*Table vs. analytic difference.*

## Sampling — bounded GGX VNDF

Chi-square sampler-vs-pdf verification (Eto & Tokuyoshi 2023 bounded sampler);
red pixels in NaN sweeps mark non-finite output — none present.

![vndf nan sweep](unit-test/output/ggx_vndf_nan_sweep.png)
*Plain VNDF sampler NaN sweep.*

![bounded nan sweep iso](unit-test/output/ggx_bounded_nan_sweep_iso.png)
*Bounded sampler NaN sweep, isotropic.*

![bounded nan sweep aniso](unit-test/output/ggx_bounded_nan_sweep_aniso.png)
*Bounded sampler NaN sweep, anisotropic.*

![bounded chi2](unit-test/output/ggx_bounded_chi2_residual.png)
*Chi-square residual, isotropic — noise only, no structured bias.*

![bounded chi2 aniso](unit-test/output/ggx_bounded_chi2_residual_aniso.png)
*Chi-square residual, anisotropic.*

![bounded pdf integral](unit-test/output/ggx_bounded_pdf_integral_heatmap.png)
*Pdf sphere-integral — uniform 1.0 after the below-horizon support fix.*

![ggx footprint isotropic](unit-test/output/ggx_footprint_isotropic_control.png)
*Sampler support footprint, isotropic control.*

![ggx footprint anisotropic](unit-test/output/ggx_footprint_anisotropic.png)
*Sampler support footprint, anisotropic.*

![dielectric specular footprint](unit-test/output/dielectric_specular_footprint.png)
*Dielectric specular lobe support footprint.*

## EON diffuse lobe

![eon disk lambert](unit-test/output/eon_brdf_disk_lambert.png)
*EON BRDF disk at σ=0 — reduces to Lambert.*

![eon disk rough](unit-test/output/eon_brdf_disk_rough.png)
*EON BRDF disk at high roughness.*

![eon materialx diff](unit-test/output/eon_materialx_diff_disk.png)
*Difference vs. an independent MaterialX EON port.*

![eon energy comp](unit-test/output/eon_energy_comp_heatmap.png)
*EON energy-compensation term.*

![eon fit error](unit-test/output/eon_energy_comp_fit_error_heatmap.png)
*Energy-compensation fit error.*

![eon furnace](unit-test/output/eon_furnace_heatmap.png)
*EON furnace (energy conservation) over roughness × incidence.*

![diffuse chi2](unit-test/output/diffuse_chi2_residual.png)
*Diffuse sampler chi-square residual.*

## Glossy-diffuse lobe

![glossy diffuse smooth](unit-test/output/glossy_diffuse_disk_smooth.png)
*BRDF disk, smooth coat.*

![glossy diffuse medium](unit-test/output/glossy_diffuse_disk_medium.png)
*BRDF disk, medium roughness.*

![glossy diffuse rough](unit-test/output/glossy_diffuse_disk_rough.png)
*BRDF disk, rough.*

![glossy diffuse aniso](unit-test/output/glossy_diffuse_disk_anisotropic.png)
*BRDF disk, anisotropic.*

![glossy diffuse furnace r x incidence](unit-test/output/glossy_diffuse_furnace_roughness_incidence.png)
*Furnace heatmap, roughness × incidence.*

![glossy diffuse furnace r x ior](unit-test/output/glossy_diffuse_furnace_roughness_ior.png)
*Furnace heatmap, roughness × IOR.*

![glossy diffuse furnace r x aniso](unit-test/output/glossy_diffuse_furnace_roughness_anisotropy.png)
*Furnace heatmap, roughness × anisotropy.*

## Metallic lobe

![metallic smooth](unit-test/output/metallic_disk_smooth.png)
*BRDF disk, smooth.*

![metallic medium](unit-test/output/metallic_disk_medium.png)
*BRDF disk, medium roughness.*

![metallic rough](unit-test/output/metallic_disk_rough.png)
*BRDF disk, rough.*

![metallic aniso](unit-test/output/metallic_disk_anisotropic.png)
*BRDF disk, anisotropic.*

![metallic furnace r x incidence](unit-test/output/metallic_furnace_roughness_incidence.png)
*Furnace heatmap, roughness × incidence.*

![metallic furnace r x aniso](unit-test/output/metallic_furnace_roughness_anisotropy.png)
*Furnace heatmap, roughness × anisotropy.*

![metallic furnace incidence x color](unit-test/output/metallic_furnace_incidence_color.png)
*Furnace heatmap, incidence × base-color tint.*

## Fuzz lobe (Zeltner LTC sheen)

![zeltner lut](unit-test/output/zeltner_lut_r_channel.png)
*Zeltner LTC coefficient LUT (R channel).*

![fuzz eval nan](unit-test/output/fuzz_eval_nan_sweep.png)
*Evaluate NaN sweep — red would mark non-finite output.*

![fuzz nan bneg](unit-test/output/fuzz_nan_sweep_bneg.png)
*Sampler NaN sweep, fuzzB < 0.*

![fuzz nan bzero](unit-test/output/fuzz_nan_sweep_bzero.png)
*Sampler NaN sweep, fuzzB = 0.*

![fuzz nan bpos](unit-test/output/fuzz_nan_sweep_bpos.png)
*Sampler NaN sweep, fuzzB > 0 — includes below-horizon incidence.*

![fuzz chi2](unit-test/output/fuzz_chi2_residual.png)
*Chi-square residual, negative fuzzB.*

![fuzz chi2 bpos](unit-test/output/fuzz_chi2_residual_bpos.png)
*Chi-square residual, positive fuzzB.*

![fuzz pdf integral](unit-test/output/fuzz_pdf_integral_heatmap.png)
*Pdf sphere-integral — uniform 1.0 across incidence × fuzzA.*

![fuzz furnace](unit-test/output/fuzz_furnace_heatmap.png)
*Fuzz furnace heatmap.*

## Transmission lobe

Rough-dielectric BTDF with VNDF refraction and TIR fallback — verification
found and fixed a half-vector-Jacobian sign bug and a matched-IOR NaN.

![transmission smooth](unit-test/output/transmission_disk_smooth.png)
*BTDF disk, smooth — tight Snell-bent lobe.*

![transmission medium](unit-test/output/transmission_disk_medium.png)
*BTDF disk, medium roughness.*

![transmission rough](unit-test/output/transmission_disk_rough.png)
*BTDF disk, rough — frosted-glass spread.*

![transmission dense](unit-test/output/transmission_disk_dense.png)
*BTDF disk, high IOR.*

![transmission eval enter](unit-test/output/transmission_eval_sweep_enter.png)
*Evaluate sweep, entering the medium.*

![transmission eval exit](unit-test/output/transmission_eval_sweep_exit.png)
*Evaluate sweep, exiting the medium.*

![transmission eval matched](unit-test/output/transmission_eval_sweep_matched_ior.png)
*Evaluate sweep at matched IOR — the case that used to NaN.*

![transmission chi2 enter](unit-test/output/transmission_chi2_residual_enter.png)
*Chi-square residual, entering rays.*

![transmission chi2 exit](unit-test/output/transmission_chi2_residual_exit.png)
*Chi-square residual, exiting rays.*

![transmission furnace](unit-test/output/transmission_furnace_heatmap.png)
*Transmission furnace heatmap.*

![transmission dispersion](unit-test/output/transmission_dispersion_spread.png)
*Dispersion — per-wavelength IOR spread.*

![transmission nan enter](unit-test/output/transmission_nan_sweep_enter.png)
*NaN sweep, entering.*

![transmission nan exit](unit-test/output/transmission_nan_sweep_exit.png)
*NaN sweep, exiting.*

![transmission nan exit diamond](unit-test/output/transmission_nan_sweep_exit_diamond.png)
*NaN sweep, exiting at diamond IOR.*

![transmission nan matched](unit-test/output/transmission_nan_sweep_matched_ior.png)
*NaN sweep at matched IOR.*

## Thin-film interference

![thinfilm disk](unit-test/output/thinfilm_disk_iridescent.png)
*Iridescent BRDF disk with thin-film active.*

![thinfilm thickness x incidence](unit-test/output/thinfilm_iridescence_thickness_incidence.png)
*Iridescent color over thickness × incidence.*

![thinfilm thickness x film ior](unit-test/output/thinfilm_iridescence_thickness_filmior.png)
*Iridescent color over thickness × film IOR.*

![thinfilm furnace](unit-test/output/thinfilm_furnace_thickness_incidence.png)
*Energy boundedness over thickness × incidence.*

![thinfilm nan sweep](unit-test/output/thinfilm_nan_sweep_grazing_kappa.png)
*NaN sweep at grazing incidence over conductor kappa.*

## Multi-lobe mixture

Pairwise chi-square residuals — standardized (observed − expected)/√E over a
sphere-bin grid, one image per incidence angle (cosθ 0.15 / 0.5 / 0.9). This
suite caught the TIR-reflect-pdf double-counting bug against specular.

![diffuse+specular ct15](unit-test/output/mixture_diffuse_specular_chi2_ct15.png)
*Diffuse + specular, grazing.*

![diffuse+specular ct50](unit-test/output/mixture_diffuse_specular_chi2_ct50.png)
*Diffuse + specular, mid incidence.*

![diffuse+specular ct90](unit-test/output/mixture_diffuse_specular_chi2_ct90.png)
*Diffuse + specular, near normal.*

![coat+specular ct15](unit-test/output/mixture_coat_specular_chi2_ct15.png)
*Coat + specular, grazing.*

![coat+specular ct50](unit-test/output/mixture_coat_specular_chi2_ct50.png)
*Coat + specular, mid incidence.*

![coat+specular ct90](unit-test/output/mixture_coat_specular_chi2_ct90.png)
*Coat + specular, near normal.*

![fuzz+diffuse ct15](unit-test/output/mixture_fuzz_diffuse_chi2_ct15.png)
*Fuzz + diffuse, grazing.*

![fuzz+diffuse ct50](unit-test/output/mixture_fuzz_diffuse_chi2_ct50.png)
*Fuzz + diffuse, mid incidence.*

![fuzz+diffuse ct90](unit-test/output/mixture_fuzz_diffuse_chi2_ct90.png)
*Fuzz + diffuse, near normal.*

![transmission+specular ct15](unit-test/output/mixture_transmission_specular_chi2_ct15.png)
*Transmission + specular, grazing — the pairing that exposed the TIR pdf bug.*

![transmission+specular ct50](unit-test/output/mixture_transmission_specular_chi2_ct50.png)
*Transmission + specular, mid incidence.*

![transmission+specular ct90](unit-test/output/mixture_transmission_specular_chi2_ct90.png)
*Transmission + specular, near normal.*

![kitchen sink ct15](unit-test/output/mixture_kitchen_sink_chi2_ct15.png)
*All lobes live at once, grazing.*

![kitchen sink ct50](unit-test/output/mixture_kitchen_sink_chi2_ct50.png)
*All lobes, mid incidence.*

![kitchen sink ct90](unit-test/output/mixture_kitchen_sink_chi2_ct90.png)
*All lobes, near normal.*

## Infrastructure checks

![noise chain](unit-test/output/noise_histogram_chain.png)
*RNG uniformity along a single sample chain (u64 stream).*

![noise spatial](unit-test/output/noise_histogram_spatial.png)
*RNG decorrelation across pixels.*

![shading frame error](unit-test/output/shading_frame_orthogonality_error.png)
*Shading-frame orthogonality error across the sphere of normals.*
