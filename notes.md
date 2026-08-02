# Research directions — notes

Ideas for the path tracer rewrite, weighted toward what the spectral + curved-transport
+ Slang autodiff combination makes unusually cheap.

## Adopt early (painful to retrofit)

1. **Separate "sample a path" from "evaluate a path".** Keep a compact vertex record
   (hit prim + barycentrics, event type, λ, random-number offsets) and a
   `[Differentiable]` function that re-evaluates β from it. This one split is the
   prerequisite for most of the list below: manifold solves, ReSTIR PT shift mappings,
   MCMC, path replay backprop, and the path inspector.

2. **Replayable sampler.** The dimension-tracking sampler should support replaying from
   a seed and perturbing by dimension. That gives primary-sample-space MLT and ReSTIR
   random-replay shifts almost for free.

3. **Autodiff hygiene on every interface.** Slang constraints worth designing around:
   - Mark `I_Spectrum`, BSDF eval/pdf, `I_Medium`, `I_DensityField` and the transport
     layers `[Differentiable]` from the start.
   - Ray queries and buffer loads are not differentiable, so reconstruct hits
     differentiably from the stored prim + barycentrics.
   - Reverse-mode loops need `[MaxIters]`.
   - Decide per `sample()` whether the pdf is attached or detached.

4. **Evaluable pdfs in media (null-collision path integral, Miller 2019).** Treat null
   collisions as real vertices with majorants exposed on `I_Medium`. Delta tracking then
   has a pdf, which hero-wavelength spectral MIS in chromatic media, ReSTIR in volumes,
   and differentiable volumes all need.

5. **A generic two-point connection solver.** The caustics plan (Snell constraints) and
   "NEE through a bending layer" are the same problem: Newton on a constraint chain with
   autodiff Jacobians. Design one solver interface that takes a transport stack, not a
   glass-specific one.

## Future targets

| Target | Why it suits this renderer |
|---|---|
| **Lensed NEE through Kerr / heat haze** | Solve the geodesic boundary-value problem to a light; forward-mode through the integrator gives the generalized geometry term. Multiple images are multiple roots, as in specular manifold sampling. The least-explored item here. |
| **∇n from `I_DensityField` via forward-mode** | Heat haze needs refractive-index gradients; autodiff of the procedural field gives them, plus SDF normals, with no finite differences. Small and immediate. |
| **Ray differentials through any transport stack** | Forward-mode over `Schwarzschild<HeatHaze<…>>` gives texture filtering footprints under lensing. |
| **Transient / time-of-flight rendering** | `cameraTime` and `pathLength` are already carried; a path-length histogram AOV gives light echoes around the hole. |
| **Inverse rendering in the viewer** | Path replay backprop (Vicini 2021) fits wavefront with constant memory; fit materials, density-field parameters or black-hole spin to a target image. Differential ratio tracking for volumes. |
| **ReSTIR DI → GI → PT** | The best payoff for "real-time-ish"; depends on items 1, 2 and 4. |
| **Langevin / Hessian-Hamiltonian MCMC** | Gradient-driven mutations using autodiff of the path contribution; strong on caustics and lensed light. |
| **Fluorescence and polarization** | The per-vertex g-shift already generalizes to wavelength-changing vertices. Polarization means a Mueller-valued β, and under Kerr it needs parallel transport, which is rare in graphics. |
| **In-shader learned sampling** | Neural radiance caching or online vMF guiding, trained with Slang autodiff. |

Deferred: visibility-discontinuity derivatives (warped-area or projective sampling).
They only matter for geometry optimization and are the heaviest lift.

## Recommendation

Do items 1–3 as part of the current infrastructure list; they cost little now and
nothing else works without them. Then make the generic connection solver the first
research feature, with glass caustics as the test case and lensed NEE as the payoff.
The ∇n trick is a good warm-up for forward-mode since the media stack is the current
work.
