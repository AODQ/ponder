# VDB volume rendering: references

Citations for the physically-based techniques used in `util-vdb.glsl` and
the volumetric path in `app/viewer/shaders/util-raytrace.glsl`. One
sentence per reference; see the shader comments at each call site for
exactly how each is applied.

- **Jendersie & d'Eon, "An Approximate Mie Scattering Function for Fog and
  Cloud Rendering" (SIGGRAPH 2023 Talk)** — the Henyey-Greenstein + Draine
  phase-function mixture and its droplet-diameter-fitted parameters
  (`utilVdbPhaseDraine`, `utilVdbPhaseFogDefaultParams`); already landed
  before this pass.
- **Kutz, Habel, Li & Novák, "Spectral and Decomposition Tracking for
  Rendering Heterogeneous Volumes" (SIGGRAPH 2017)** — the chromatic/RGB
  weighted delta- and ratio-tracking scheme (`utilVdbDeltaTrack`,
  `utilVdbRatioTrack`): a shared achromatic majorant drives the free-flight
  walk, and a per-channel importance-sampling correction weight keeps each
  color channel's estimator unbiased despite sharing one walk.
- **Novák, Georgiev, Hanika & Jarosz, "Monte Carlo Methods for Volumetric
  Light Transport Simulation" (EG STAR / SIGGRAPH course notes, 2018)** —
  general reference for the ratio-tracking transmittance estimator and the
  weighted-delta-tracking correctness argument (apply the true/proxy
  probability ratio at *every* step, real or null, not just the final one).
- **Pharr, Jakob & Humphreys, *Physically Based Rendering* (pbrt), the
  `HomogeneousMedium`/`GridMedium` chapter** — the `sigma_a`/`sigma_s`
  (absorption/scattering) parameterization replacing the old scalar
  `sigma` + single albedo vector, and blackbody emission from a
  temperature field as the standard way volume renderers add self-emission.
- **Wyman, Sloan & Shirley, "Simple Analytic Approximations to the CIE XYZ
  Color Matching Functions" (JCGT 2(2):1-11, 2013)** — the closed-form
  multi-Gaussian fit to the CIE 1931 2-degree standard observer
  (`utilCieXyz1931`) used to convert an in-shader Planck's-law spectral
  radiance integral to RGB without a lookup texture (`utilBlackbodyRadiance`);
  coefficients verified against the paper's own supplemental source
  (`curves/multiLobeFit1931.cpp`, the
  `JournalOfComputerGraphicsTechniques/TEST-0002-02-01-Wyman-Sloan-Shirley`
  GitHub repo), not transcribed from memory.
- **IEC 61966-2-1 (the sRGB standard)** — the CIE XYZ to linear sRGB (D65
  white point) 3x3 matrix `utilBlackbodyRadiance` uses to finish the
  temperature-to-color conversion.
- **Fong, Wrenninge, Kulla & Habel, "Production Volume Rendering" (SIGGRAPH
  2017 course)** — general production-renderer precedent for exposing
  absorption/scattering/emission as independent, artist-facing volume
  parameters, and for driving emission off a simulation temperature grid.
