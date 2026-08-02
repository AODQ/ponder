# Path infrastructure — implementation plan

Scope: three pieces of integrator infrastructure, done before further media or light
sampling work, because ReSTIR, MLT, the connection solver, differentiable evaluation and
the path debugger all build on them:

1. sampler API cleanup
2. separating "sample a path" from "evaluate a path" (consumers)
3. replayable sampler (seeded replay, primary sample space)

Each step must leave the rendered image unchanged, except where noted, so a known-good
render is the regression check throughout.

---

## 1. What already exists

| piece | where | state |
|---|---|---|
| stateless sampler, addressed by (pixel, sample index, dimension) | `sample/sampler.slang` | the property everything here depends on |
| per-bounce dimension layout | `BounceDimension`, `bounceDimensionOffset` | X/Y aliases reserve 2D slots by hand |
| unbounded streams | `sample/sample-stream.slang`, `nextF32WithIndex` | (dimension, index) rehash; index 0 is the normal draw |
| dimension reuse check | `types/debug-path.slang`, `ponder/debug-path.cpp` | logs each dimension, flags duplicates |
| path record | `DebugPathVertex` | close to the vertex record below; lacks prim, barycentrics, pdfs, sampler offsets |
| integrator | `entry/wavefront-integrate.slang` | sampling and evaluation interleaved in one loop |

---

## 2. Sampler API cleanup

Goal: short call sites, dimensions that know their width, one entry point for streams,
and a swappable source of random numbers.

```slang
let bounceSampler = sampler.bounce(bounce);
let ξ: f32v2 = bounceSampler.next2(BounceDimension.Direction);
var stream = bounceSampler.stream(BounceDimension.MediumDistance);
```

- **Bounce view.** `sampler.bounce(bounce)` computes the bounce offset once.
- **Dimension widths.** Each dimension declares 1D or 2D; the X/Y aliases go away. In the
  debug build, `next1` / `next2` check that a dimension is used at its declared width.
- **Streams from the same object.** `stream(dimension)` returns the unbounded stream; one
  logging rule (only index 0 is logged).
- **`I_SampleGenerator`.** The hash is one implementation. Call sites only ask for
  "dimension k of this path", so other generators drop in later (step 4, and Sobol or
  blue noise as a separate experiment).
- **Direction slot.** `BsdfDirection` becomes `Direction`: surface and medium vertices are
  mutually exclusive per bounce, so they share it.

Check: identical image apart from which random number each draw receives (the dimension
layout changes), so compare converged images, not single samples.

---

## 3. Sample / evaluate split

Goal: sampling produces vertices; a consumer decides what to do with them. Separation is
in the API, not in storage, so the default path costs nothing extra.

```slang
public struct PathVertex {
	TransportEvent event;
	HitModel model;          // prim + barycentrics, valid for surfaces
	f32v3 x;
	f32v3 ωᵢ;
	f32v3 ωₒ;
	Spectrum weight;         // transport weight (spectralWeight) into this vertex
	f32 pdfDirection;        // of ωₒ, for mis
	f32 pdfLight;            // later, nee
	u32 dimensionOffset;     // where this bounce's random numbers start
	// medium vertices: medium index / phase parameters
};

public interface I_PathConsumer {
	[mutating] func vertex(vertex: PathVertex) -> bool;   // false ends the path
	[mutating] func escape(vertex: PathVertex);
}

func samplePath<C : I_PathConsumer>(
	inout I_TraceTransport transport,
	sampler: Sampler,
	inout C consumer
);
```

Consumers:

- **`PathConsumer_Evaluate`**: accumulates β and Lᵢ as it goes; this is today's integrator.
  No storage; slang specializes it, so it costs what the loop costs now.
- **`PathConsumer_DebugRecord`**: writes vertices to a buffer. Replaces the debug recorder; the
  `if (debugPathActive())` blocks leave the integrator.
- **`EvaluateFromRecord`**: re-evaluates a recorded path. Later the base for
  differentiable evaluation, mis re-weighting and shift mappings.

Notes:

- The vertex carries pdfs from the start, because nee + mis needs them.
- Russian roulette stays in sampling; its probability is part of the vertex weight.
- Curved transport: store chord points with the record rather than re-integrating
  geodesics during re-evaluation.

Check: `PathConsumer_Evaluate` reproduces the current image bit-for-bit for the same seeds.

---

## 4. Replayable sampler

**Seeded replay (ReSTIR random replay).** A pixel replays a neighbour's path with the
neighbour's random numbers.

- `Sampler(seed)` alongside `Sampler(pixel, sampleIndex)`; same dimension addressing.
- Check: replaying a path with its own seed reproduces it exactly (the debug path shows
  identical vertices).

**Primary sample space (PSSMLT).** A Markov chain mutates the random numbers.

- A `PrimarySampleGenerator : I_SampleGenerator` returns the chain's stored value for a
  dimension, or draws and stores a fresh one (lazy, Kelemen et al. 2002).
- Every (dimension, stream index) maps to a fixed slot. Streams get a capped number of
  slots; beyond that they fall back to the hash, so heavy delta tracking is only partly
  mutable.
- Mutations: small perturbations per dimension, plus large steps that redraw everything.
- PSSMLT itself is a separate integrator mode built from these pieces; it is not part of
  this plan.

---

## 5. Autodiff readiness

From the research notes, scoped down. `[Differentiable]` costs nothing at runtime when
unused, but every marked function must be differentiable or annotate `no_diff`, which
slows compiles and constrains code still in flux.

- Mark only the pure math now: `I_Spectrum` operations, phase functions, bsdf eval / pdf,
  `I_Medium.at`.
- Transports and the integrator wait for the first differentiable feature. Ray queries
  and buffer loads are not differentiable; re-evaluation reconstructs hits from the
  recorded prim + barycentrics.
- Loops that reverse-mode will run through need `[MaxIters]`.
- Every `sample()` decides whether its pdf is attached (differentiated) or detached.

Enforced by a check build, so the normal build never pays for it:

- `include/differentiable.slang` defines `PonderDifferentiable` (and
  `PonderDifferentiableMethod`, which adds `[NoDiffThis]`). They expand to the attributes
  only when `PONDER_AUTODIFF_CHECK` is defined, and to nothing otherwise.
- `entry/autodiff-check.slang` calls `fwd_diff` / `bwd_diff` on every marked function;
  marking alone generates nothing, so only listed functions are verified.
- `cmake --build … --target ponder-autodiff-check` compiles it with warnings as errors.
  It is not part of the default build or viewer startup.

---

## 6. Order

1. Sampler API cleanup (§2). **Done**, except the debug-build width check.
2. Sample / evaluate split (§3). **Done**: `PathVertex`, `I_PathConsumer`,
   `PathConsumer_Evaluate`, `PathConsumer_DebugRecord`. `PathVertex` still lacks the
   direction pdf and the bounce's first dimension; `EvaluateFromRecord` waits for a user.
3. Seeded replay (§4, first half). **Done**: `Sampler(seed)`, `getSeed()`.
4. Primary sample generator (§4, second half). **Deferred**, alongside the caustics work.
5. Autodiff readiness (§5), when the first differentiable feature starts.

Then resume `plan-media-stack.md` phase 1 (scattering is wired through `samplePath`),
then nee + mis, then ReSTIR (uses step 3).

## 7. After this plan

- **A generic two-point connection solver.** Glass caustics (Snell constraints) and nee
  through a bending layer are the same problem: Newton on a constraint chain with
  autodiff Jacobians. One solver interface that takes a transport stack, not a
  glass-specific one. Glass caustics first, lensed nee later.
- **Transient (time-of-flight) rendering.** `cameraTime` and `pathLength` are already
  carried; a path-length histogram output is nearly free.
