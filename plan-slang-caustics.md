# Differentiable caustics — manifold sampling written in Slang

A learning project. The goal is to get hands-on with **automatic differentiation**
(autodiff) by using it for something the renderer genuinely cannot do today:
**caustics** — the bright focused patterns light makes after passing through glass.

Written for someone who has never used autodiff before. Every term is explained the
first time it appears. Timeline is deliberately loose; the stages are ordered so that
each one is worth having even if the next never happens.

---

## 0. The one-paragraph version

Our path tracer cannot render caustics from small lights, for a structural reason
explained in §1. There is a known fix (§2) that involves solving a small equation with
Newton's method. Newton's method needs derivatives, and in every existing
implementation of this technique those derivatives are worked out **by hand**, on
paper, and then typed in — which is both the bulk of the work and the documented reason
one production attempt failed (§4). Autodiff computes those derivatives mechanically
from the code itself. So: implement the technique, but let the compiler do the calculus.

---

## 1. Why caustics are hard

Our path tracer finds light two ways at each surface point it hits:

1. **BSDF sampling** — pick a direction according to the material, trace, see what's
   there. If it happens to land on a light, great.
2. **NEE** (next-event estimation) — draw a line straight from the surface point to a
   point on a light and check whether anything blocks it.

NEE is what makes small lights usable. Without it, BSDF sampling has to *randomly
stumble onto* a small light, which almost never happens, so the image is extremely
noisy.

Now put a glass ball between the floor and the light. To carry light from the lamp to a
spot on the floor, a path has to bend through the glass twice. Those bends are not
free choices — they are forced by Snell's law.

- NEE fails completely. It draws a **straight line** from floor to lamp. That line hits
  the glass, gets counted as blocked, and contributes nothing. NEE has no way to say
  "go from here to there *via* two refractions".
- BSDF sampling technically can find those paths, but only by randomly guessing a
  direction off the floor that, after two refractions, happens to hit the lamp. For a
  small lamp that's a needle in a haystack.

So caustics from small lights are the one thing a plain path tracer structurally cannot
sample. That is why they're missing from our renders, and it isn't a bug — it's the
sampling strategy's blind spot.

---

## 2. The fix: manifold sampling

The idea, in plain terms: **stop guessing the specular path and solve for it.**

Fix the two endpoints — a point on the floor, and a point on the lamp. Ask: where on
the glass surface must the ray bend so that it starts at the floor point and ends at
the lamp point, with both bends obeying Snell's law? That's a well-posed question with
a small number of answers, and we can solve for them numerically.

### 2.1 The constraint

Describe a candidate bend point on the glass by two numbers. We'll use the
**barycentric coordinates** `(u, v)` of a point inside a triangle, because that's
already what our ray hits report (`RayQueryResult`, `app/viewer/shaders/util-raytrace.glsl:22`).

Given `(u, v)` we can compute:

- `p(u, v)` — the 3D position of that point (linear interpolation of the triangle's
  three vertex positions)
- `n(u, v)` — the surface normal there (interpolation of the three vertex normals)

From `p` we get the incoming direction `wi` (from the floor point to `p`) and the
outgoing direction `wo` (from `p` to the lamp point). Snell's law is satisfied at `p`
only for specific relationships between `wi`, `wo`, and `n`.

Write a function that measures **how badly Snell's law is violated**:

```
C(u, v) -> (c1, c2)     // two numbers, both zero exactly when the path is valid
```

Two numbers in, two numbers out. When `C(u, v) = (0, 0)`, we have found a real light
path. The set of `(u, v)` where this holds is what the papers call the *manifold*,
hence "manifold sampling".

There are two standard ways to write `C`:

**Half-vector form.** Build the "generalized half-vector" `h = wi + eta * wo`, where
`eta` is the relative index of refraction. For a valid refraction, `h` points exactly
along the normal `n`. So build a tangent frame at `p` — two unit vectors `s` and `t`
lying in the surface, perpendicular to `n` — and set:

```
C = ( dot(h_normalized, s), dot(h_normalized, t) )
```

Both components are zero exactly when `h` has no sideways component, i.e. when it
points along `n`. This is the simpler form, and it's what to implement first.

**Angle-difference form.** Convert directions to spherical coordinates (two angles
each) and directly compare the direction light *would* refract to against the
direction we actually need it to go:

```
C = ( theta_actual - theta_ideal, phi_actual - phi_ideal )
```

More robust, more work. This is the stretch goal — see §11 for why it matters.

### 2.2 Newton's method, plainly

We have a function `C(u, v)` and we want the `(u, v)` that makes it zero. Newton's
method is the standard way:

1. Guess a starting `(u, v)`.
2. Evaluate `C` there. If it's close enough to zero, stop — we've found a path.
3. Otherwise, work out **which direction to move `(u, v)` to make `C` smaller**, step
   that way, and go back to 2.

Step 3 is the only interesting part, and it's where derivatives enter. We need to know
how each output of `C` responds to each input. That's four numbers:

```
     d c1/d u    d c1/d v
     d c2/d u    d c2/d v
```

This 2x2 table of partial derivatives is called the **Jacobian** (written `grad C`).
It's just "the slope, in all four input/output combinations". Given it, the Newton step
is:

```
(u, v)_next = (u, v) - inverse(grad C) * C(u, v)
```

Read that as: take the current error `C`, run it through the inverse slope table to
convert "how wrong the output is" into "how far to move the input", and subtract.

**The Jacobian is the thing autodiff will give us.** That's the whole hinge of this
project.

### 2.3 Reprojection

One wrinkle. The Newton step moves `(u, v)` as if the surface were flat, which it
isn't. So after each step, the new point must be put **back onto the real geometry**.
The standard fix is a ray cast: shoot a ray from the previous path vertex toward the
stepped point, and use wherever it actually hits the glass as the new vertex.

Important consequence for us: **these ray casts are not part of the derivative.** The
derivative is taken of `C` only, with the triangle's vertex data held fixed. The
differentiated region is a small pure function — roughly 50 lines with no ray casts, no
random numbers, no texture lookups in it. That is what makes this project tractable,
and it's worth re-reading that sentence before worrying about any of the rest.

---

## 3. What autodiff actually does for us

### 3.1 What autodiff is

You write a function normally. The compiler reads it and mechanically generates a
*second* function that computes the derivatives of the first, by applying the chain
rule to every individual operation you wrote. You never do calculus. You never pick a
step size. The result is exact, not an approximation.

In Slang the spelling is: put `[Differentiable]` on your function, and you may then
call `fwd_diff(yourFunction)(...)`.

### 3.2 Forward mode — the only mode this project needs

There are two flavours of autodiff:

- **Forward mode** costs one pass *per input*, and each pass gives you the derivative
  of *all outputs* with respect to that one input.
- **Reverse mode** costs one pass *per output*, and each pass gives derivatives with
  respect to *all inputs*. This is the flavour machine learning uses, because there you
  have one output (the loss) and millions of inputs.

Our `C` has 2 inputs and 2 outputs, so either mode costs two passes. **Use forward
mode** — it's much simpler to reason about, has no memory overhead, and is the easier
one to learn first.

Concretely: run `fwd_diff(C)` once with `(u, v)` seeded as "u is changing, v is
fixed" and you get the Jacobian's first column. Run it again seeded the other way and
you get the second column. Two calls, whole Jacobian, no hand calculus.

In Slang, "seeding" means wrapping each input in a `DifferentialPair` — a value paired
with its rate of change. `diffPair(u, 1.0)` means "u, and we're asking about motion in
u". `diffPair(v, 0.0)` means "v, held still".

### 3.3 The work this replaces

This is the concrete payoff, so it's worth being specific. To build that one 2x2
Jacobian, the official reference implementation (Mitsuba 2, `manifold_ss.cpp`)
hand-derives every one of these by chain rule and types them in:

```
dp_du, dp_dv      position derivatives
dn_du, dn_dv      normal derivatives
ds_du, ds_dv      tangent-frame derivatives
dt_du, dt_dv      the other tangent-frame derivatives
dwi_du, dwi_dv    incoming-direction derivatives
dwo_du, dwo_dv    outgoing-direction derivatives
dh_du, dh_dv      half-vector derivatives, composed from all of the above
```

and then assembles entries like:

```
dH_dX(0,0) = dot(v1.ds_du, h) + dot(v1.s, dh_du)
```

Every one of those lines is an opportunity for a sign error that produces a solver
which *almost* works. With `[Differentiable]`, none of them get written. You write only
the thing you actually understand — `C(u, v)` itself — and ask for its derivative.

---

## 4. Why this is worth doing (and where it's genuinely new)

The technique is not new: Jakob & Marschner 2012 (manifold exploration), Hanika et al.
2015 (MNEE), Zeltner et al. 2020 (SMS), and a steady stream since. Blender Cycles
already ships MNEE. We are not inventing the sampling theory.

What *is* new is doing it with machine-generated derivatives, and there's hard evidence
that this is the load-bearing difficulty rather than an aesthetic preference. A 2025
master's thesis integrated SMS into Cycles and reported two things:

1. **Reflective caustics: under 1% solver success rate, feature disabled.** The cause
   is a derivative blowup, not a logic error. For reflection the half-vector is
   `H = wi + wo`, and near head-on incidence `wi` and `wo` are nearly opposite, so
   `H` approaches the zero vector. Normalizing it explodes, `NaN` floods the Jacobian,
   and the solver diverges instantly. Refraction survives only because the `eta`
   factor keeps `h` from collapsing.

2. **The known remedy stalled on hand-derivation.** The angle-difference constraint is
   exactly what Zeltner proposes to fix that fragility. The thesis implemented it
   *partially* and then stopped, because the path contribution and the MIS weights all
   have to be re-derived for the new formulation — described in the thesis as "a
   mathematically intensive task that was not completed". They also had to hand-write
   derivative twins of every helper function: `ad_refract`/`ad_d_refract`,
   `ad_sphcoords`/`ad_d_sphcoords`, `ad_reflect`/`ad_d_reflect`.

Those hand-written derivative twins are precisely what `[Differentiable]` generates for
free. So the project has a real thesis behind it:

> The reason reflective manifold sampling is broken in the one production integration
> that attempted it is a hand-derivation wall. Autodiff removes exactly that wall.

If §11 works, we get reflective caustics that Cycles gave up on.

---

## 5. Architecture: a separate Slang pass, GLSL untouched

Do **not** port the path tracer to Slang, and do not put the Newton solver inside
`utilIrradianceWalk`. Instead:

```
  [existing GLSL path tracer]
        |  writes seed records to a buffer:
        |  { receiver hit point+normal, sampled light point, caster object id }
        v
  [new Slang compute pass]   <-- the only Slang code in the project
        |  runs Newton solves, writes found paths + their contributions
        v
  [existing GLSL path tracer / accumulate]
        |  reads results, adds them to the pixel
        v
      image
```

Two reasons this is the right shape:

- **No translation.** Our OpenPBR library, the path tracer, all 100-odd unit-test
  shaders stay in GLSL. Slang appears only in one new file.
- **No divergence disaster.** A Newton loop with ray casts inside it, inlined into the
  main path-tracing loop, would wreck occupancy — most pixels do zero iterations while
  a few do twenty, and on a GPU the whole warp waits. As its own dispatch over a
  compacted list of seeds, every thread has work.

---

## 6. Stage 0 — get Slang compiling at all

### 6.1 What Slang and `slangc` are

Slang is a shading language (broadly HLSL-shaped) whose compiler, **`slangc`**, turns
`.slang` source files into **SPIR-V** — the same GPU bytecode format `.spv` that
`glslangValidator` produces from our `.comp` files. Vulkan doesn't care which compiler
made the bytecode. Slang's distinguishing feature, and the reason we're here, is
built-in autodiff.

`slangc` is not installed on this machine. Get it from the shader-slang GitHub releases
(or the AUR `shader-slang` package) and confirm `slangc --version` runs.

### 6.2 The engine change, spelled out

Right now, when the engine wants a compute pipeline, `vkof` does something
delightfully blunt: it builds a **command-line string** and shells out to the compiler.
See `compile_and_load_shader_module` in `lib/vkof/src/vkof.cpp` (starts ~line 4082).
It assembles this, at line 4100:

```
glslangValidator -S comp --target-env vulkan1.3 -gVS -D<defines> -I<includes> -o <out.spv> <in.comp>
```

runs it with `popen`, reads the resulting `.spv` file back off disk, and hands the
bytes to `vkCreateShaderModule`.

So "add Slang to the engine" means: **if the shader path ends in `.slang`, build a
different command string.** Roughly:

```
slangc -target spirv -stage compute -entry main -o <out.spv> <in.slang>
```

Pass defines and include paths the same way (`-D`, `-I`). Keep everything else — the
`.spv` caching, the error capture into `sLastCompileError`, the hot-reload dependency
scan — exactly as it is. It is genuinely a small, contained change; there is no build
system work, no new library to link, no CMake surgery.

One thing to watch: `slangc` and `glslangValidator` can disagree about matrix layout
(row-major vs column-major) and about struct field packing. If numbers come out
scrambled rather than wrong-by-a-little, suspect layout before suspecting your math.

### 6.3 The two unknowns, and what "spike" means

A **spike** is a throwaway experiment whose only job is to answer one risky question
before you build anything on top of it. Thirty lines, delete it afterwards. Two
questions need spiking, because if either answer is "no" the architecture in §5 changes.

**Unknown 1 — can Slang read our geometry buffers?** Our shaders don't bind buffers as
descriptors. They pass raw 64-bit GPU addresses around and cast them, using
`GL_EXT_buffer_reference`:

```glsl
layout(buffer_reference, scalar) buffer GpuMorPositionBuffer { f32v3 data[]; };
// ...
const f32v3 pos0 = GpuMorPositionBuffer(model.positions).data[i0];
```

Slang supports this — a pointer type in Slang lowers to SPIR-V's
`PhysicalStorageBuffer64` addressing, which is the same mechanism. But "supports it"
and "agrees with our exact struct layouts" are different claims. Our GLSL declares
`scalar` block layout; Slang has its own default rules.

*Spike:* one Slang compute shader that takes a `GpuMorPositionBuffer` address as a
push constant, reads vertex 0's position, and writes it to a buffer. Compare against
the same read done from GLSL. If the three floats match, this whole risk evaporates.

**Unknown 2 — can Slang trace rays the way we do?** We cast rays from compute shaders
using **ray queries** (`GL_EXT_ray_query`): you initialize a query object against the
acceleration structure, step it in a loop, and read back hit info. See
`utilTraceRayImpl`, `util-raytrace.glsl:44-145`. The Newton solver needs this for its
reprojection step (§2.3) and for the final visibility check.

Slang spells the same hardware feature differently — an acceleration structure is a
`RaytracingAccelerationStructure` and the query is a `RayQuery<>` object with
`TraceRayInline` / `Proceed` / committed-hit accessors. Our TLAS is bound as a plain
descriptor (`resolve.comp:26`, set 0 binding 3), which Slang can declare.

*Spike:* one Slang compute shader that casts a single ray down the middle of the
screen and writes the hit distance, primitive index, and barycentrics. Compare against
the GLSL path. Use opaque-only tracing at first and skip the alpha-test loop — the
caustic pass doesn't need it.

### 6.4 Also in stage 0: a scene to look at

Two small asset chores, easy to forget until you need them:

- **A small light.** `stage::LightInstanceQuad` is parsed, gizmo-editable and saved,
  but never actually reaches the renderer — nothing converts it to geometry or an
  emitter (`lib/stage/src/stage-imgui.cpp:477` edits it; no consumer exists). Don't
  fix that for this project. Just author a small emissive triangle or quad in a glTF,
  which already works through the existing emitter list.
- **Glass with smooth normals.** Manifold sampling needs a continuous normal field —
  flat-shaded geometry has no manifold to walk on, because the normal is constant
  across each triangle and jumps at the edges. Check that whatever glass object you
  use has interpolated vertex normals. The glass dragon should be fine.

---

## 7. Stage 1 — the differentiable constraint, and how to know it's right

This stage has no renderer in it at all. It's a unit test. It is also where you
actually learn autodiff, so don't rush it.

Write the half-vector constraint from §2.1 as a Slang function:

```
[Differentiable]
float2 constraint(
    float2 uv,                        // the unknown we solve for
    no_diff TriangleData tri,         // vertex positions + normals, held fixed
    no_diff float3 receiverPoint,     // path vertex before the glass
    no_diff float3 lightPoint,        // path vertex after the glass
    no_diff float eta                 // relative index of refraction
)
```

`no_diff` marks the inputs we are *not* differentiating with respect to. It's not an
optimization — it's documentation the compiler enforces, and it keeps the generated
derivative code small.

Then get the Jacobian with two forward-mode calls, seeded as described in §3.2.

### 7.1 The gradient check

**Always do this. It is the single habit that separates a working autodiff project from
a mysterious one.** A wrong derivative throws no error and crashes nothing — the solver
just quietly converges slowly, or to the wrong place, or not at all, and you spend two
days blaming your constraint function.

The check: compare autodiff's answer against a crude finite difference.

```
analytic  = column j of the Jacobian, from fwd_diff
numeric_j = ( C(uv + h*e_j) - C(uv - h*e_j) ) / (2h)     // e_j is the unit step in input j
```

with `h` around `1e-4` in the units of your parameterization. Run it at a dozen random
`(u, v)` values on a real triangle and compare.

**What agreement to expect.** We're in 32-bit float, so don't demand 1e-7. Relative
agreement of about `1e-3` is a pass. And that's fine, because derivative bugs are
almost never subtle: a wrong derivative is wrong by a **sign**, a **factor of two**, or
is **exactly zero**. You are looking for those, not for the last digit.

Zero deserves special mention: if a derivative comes back exactly `0.0` and you didn't
expect it, the usual culprit is a `clamp`, `min`, `max`, or `saturate` sitting in the
chain. Those have zero slope once they're pinned at their limit, so they silently kill
the gradient. Our shaders clamp very liberally, so expect to meet this.

---

## 8. Stage 2 — the Newton solver

Now the solver, as its own Slang compute pass (§5). Per seed record:

```
uv = initial guess
repeat up to maxIterations:
    C = constraint(uv, ...)
    if length(C) < threshold:  converged, done
    J = jacobian(uv, ...)              // two fwd_diff calls
    if abs(determinant(J)) < 1e-6:     bail out, this seed is hopeless
    step = inverse(J) * C
    propose uv' by stepping and reprojecting onto the geometry via a ray cast (§2.3)
    if the reprojection ray missed the glass:  halve the step, try again
    else:  accept, and allow the step to grow back (double it, capped at full size)
```

Three safeguards there are lifted straight from the reference implementation and are
not optional:

- **Determinant guard.** A near-singular Jacobian means the inverse is garbage.
  Mitsuba bails at `1e-6`. Without this you get `inf` in your step.
- **Backtracking.** Halve the step when the reprojection ray misses the object; double
  it back (capped) when it lands. Pure Newton overshoots constantly on curved glass.
- **Iteration cap.** Non-convergence is normal and expected — most seeds fail, and
  that's fine, it's a *sampling* technique. Just bound the work.

### 8.1 Seeding

Where does the initial guess come from? Two parts:

1. **Find the caster.** Cast a probe ray from the receiver point toward the light and
   see what glass object it hits. That's your caster.
2. **Randomize.** Pick a random point on that object's primitives as the starting
   `(u, v)`. Different random starts find different solutions — which matters, because
   a caustic generally has several valid paths per pixel, and Newton finds at most one
   per run.

You'll also want a **"caustic caster" flag** on materials or instances, so the solver
is only attempted for objects where it makes sense. Both Mitsuba and Cycles have one
for exactly this reason. A bool in the stage file is enough.

Success criterion for this stage: given a receiver point and a light point, the solver
lands on a bend point on the glass that actually satisfies Snell's law, verified
independently. No pixels yet.

---

## 9. Stage 3 — turning a found path into light

A found path isn't a contribution yet. The full weight needs:

- the BSDF values at each vertex (we have these),
- the emitter radiance (we have this),
- a visibility check along the completed path (ray casts, no derivatives),
- and a **generalized geometric term** `G`, which accounts for how much the path
  "spreads" — the specular-path analogue of the usual `cos/r^2` factor in NEE.

`G` is itself a determinant of a Jacobian — the derivatives of the constraints across
all three vertices, assembled into a block system. So autodiff supplies this too, with
the same machinery. Mitsuba clamps `G` at 10 to limit variance, which is the same
instinct as our existing firefly clamp; do likewise.

**Use the biased estimator first.** Run a fixed budget of solver attempts, collect the
unique solutions you found, and weight each by one over that count. This is biased —
it systematically misses solutions the budget didn't reach — but it's simple and its
cost is bounded. §10 fixes the bias.

This is the stage where caustics appear on screen.

---

## 10. Stage 4 — unbiased estimator, and MIS

### 10.1 The Bernoulli trick, explained

To be unbiased, the estimator must divide by `p`, the probability that a random seed
leads the solver to *this particular* solution. We cannot compute `p` — it depends on
the shape of the solver's basin of convergence, which has no closed form.

The trick: we don't need `p`, we need `1/p`, and there's a way to estimate that without
ever knowing `p`. Keep re-running the solver from fresh random seeds and count how many
attempts it takes to land on the same solution again. For a repeated trial with
probability `p`, the expected number of attempts until success is exactly `1/p`. So the
count itself is an unbiased estimate of `1/p` — no formula required.

That's the Bernoulli-trial estimator. It is unbiased and it costs many extra solver
runs per sample, which is exactly why §9 starts biased.

### 10.2 MIS

Right now caustic contributions would simply replace NEE for that light sample. Better
is to MIS-combine them with the existing NEE and BSDF-sampling estimators, so that
roughness and lighting changes degrade gracefully instead of switching over abruptly.
Note that the Cycles integration skipped this, and Zeltner recommends it.

---

## 11. Stage 5 (stretch) — angle-difference constraint, reflective caustics

This is the part that would be genuinely new, and it follows directly from §4:

- Swap the half-vector constraint for the angle-difference constraint (§2.1). It
  doesn't collapse at head-on incidence, so it should not produce the `NaN` Jacobian
  that killed reflective caustics in Cycles.
- Re-derive the path contribution and MIS weights for the new formulation. This is the
  step the thesis abandoned as too mathematically intensive — and it's a Jacobian
  determinant, so it's autodiff's job, not yours.

**One free bug, already paid for.** The angle-difference constraint has an azimuth
periodicity trap: angles near `2*pi` must be wrapped into `[-pi, pi]` or the solver
will refuse to converge for no visible reason. This was a real bug in the official SMS
implementation, fixed in 2021. Wrap the angles from the start.

If this works: reflective caustics via manifold sampling, which the one production
integration that tried it could not get above a 1% success rate.

---

## 12. Pitfalls, with reasons

| pitfall | why it bites | what to do |
|---|---|---|
| `H = wi + wo -> 0` for reflection | normalizing a near-zero vector sends `inf`/`NaN` into the Jacobian; documented sub-1% success rate in Cycles | refraction only until §11 |
| `normalize()` and `sqrt()` at zero | their derivatives are `NaN` even where the value is fine | guard lengths before normalizing inside the differentiated function |
| `clamp` / `min` / `max` / `saturate` | derivative is exactly zero once pinned, silently killing the gradient | keep them out of the differentiated path, or expect zeros and recognise them |
| near-singular Jacobian | `inverse()` produces garbage | bail at `abs(det) < 1e-6` |
| pure Newton steps | overshoot badly on curved glass | halve on reprojection miss, double back on success |
| unclamped geometric term | fireflies | clamp `G` (Mitsuba uses 10) |
| flat-shaded geometry | no continuous normal field means no manifold | require smooth normals on casters |
| azimuth wrap | solver silently fails to converge | wrap to `[-pi, pi]` (§11) |
| reprojection strategy | ray-cast vs direct projection onto the triangle changes robustness near edges; the thesis flags this as a real difference from Mitsuba | start with ray casts, keep it swappable |
| Slang/GLSL struct layout | scrambled values rather than wrong ones | settle it in the §6.3 spike, not later |

---

## 13. Validation

Furnace tests and chi-square tests cannot see this — they test BSDFs, and this is a
sampling strategy. The right instrument is the one Zeltner's repo ships: a **reference
integrator restricted to the same path class**.

Concretely: an integrator that accumulates *only* paths of the form
`eye -> ... -> diffuse -> refraction(s) -> light`, found by brute-force BSDF sampling
at very high sample counts. Render the same scene with caustic sampling on, difference
the two images, and the difference should go to zero as both converge. Same shape as
the existing MaterialX crosscheck heatmaps.

That's an expensive render, so it belongs behind `--thorough` via the
`test::statistical_*` helpers, per the standing rule that default test runs stay cheap
on the GPU.

---

## 14. Definition of done — with fallback wins

Staged so that stopping early still leaves something real:

- **Stage 0 done** = a `.slang` compute shader compiles through `vkof`, reads one of
  our geometry buffers correctly, and casts one ray correctly. *This alone unlocks
  every other autodiff project we discussed.*
- **Stage 1 done** = a differentiable Snell's-law constraint whose autodiff Jacobian
  matches finite differences. *This is the autodiff learning goal, fully met, with a
  test to prove it.*
- **Stage 2 done** = a Newton solver that reliably finds valid refractive paths through
  the glass dragon. *A legitimate finish even if no pixel ever lights up.*
- **Stage 3 done** = caustics in the render. *The screenshot.*
- **Stage 4 done** = unbiased, MIS-combined, defensible.
- **Stage 5 done** = reflective caustics, i.e. the thing nobody has landed this way.

Stages 0-2 are the load-bearing ones. Everything after is upside.

---

## 15. House rules that apply

- Bracket edited chunks in `lib/` and `app/` with `(TODO REVIEW)` before and after.
  `unit-test/` is exempt.
- Comments 1-3 lines, always on their own line, never trailing.
- Build and test against `build/release`, not `build/debug`.
- Heavy render sweeps go behind `--thorough`.
- Note for the `.slang` file: it lives in `app/viewer/shaders/` alongside the `.comp`
  files, and its `.spv` lands in `shaders/bin/` like everything else.

---

## 16. References

**The technique**

- Zeltner, Georgiev, Jakob. *Specular Manifold Sampling for Rendering High-Frequency
  Caustics and Glints.* SIGGRAPH 2020.
  https://tizianzeltner.com/projects/Zeltner2020Specular/
- Reference implementation (Mitsuba 2). The file to read is
  `src/librender/manifold_ss.cpp` — both constraint forms, the hand-derived
  derivatives, the Newton safeguards, and the geometric term are all in there.
  https://github.com/tizian/specular-manifold-sampling
- Hanika, Droske, Fascione. *Manifold Next Event Estimation.* Computer Graphics Forum
  34(4), 2015. The deterministic predecessor; this is what Cycles ships.
  https://onlinelibrary.wiley.com/doi/10.1111/cgf.12681
- Jakob, Marschner. *Manifold Exploration.* SIGGRAPH 2012. The origin of the
  constraint formulation.

**The integration experience report — read this one**

- Delattre. *Integrating Specular Manifold Sampling in Cycles Renderer for Enhanced
  Specular Light Transport.* Uppsala University, July 2025. Section 4.7 is the list of
  everything that went wrong in a real production integration, including the reflective
  half-vector blowup and the abandoned angle-difference derivation.
  https://uu.diva-portal.org/smash/get/diva2:1985010/FULLTEXT02.pdf

**Where the field went next (context, not prerequisites)**

- Fan et al. *Manifold Path Guiding for Importance Sampling Specular Chains.*
  SIGGRAPH Asia 2023. Learns seed distributions; up to 40x variance reduction.
  https://arxiv.org/html/2311.12818v1
- *Specular Polynomials.* SIGGRAPH 2024. Replaces Newton with polynomial resultants —
  deterministic, finds all one-bounce solutions, GPU-friendly.
  https://arxiv.org/abs/2405.13409
- *Interactive Rendering of Caustics using Dimension Reduction for MNEE.* i3D 2024.
  Precedent for doing this at interactive rates on a GPU.
  https://dl.acm.org/doi/10.1145/3651297

**Slang**

- Automatic Differentiation user guide — `[Differentiable]`, `fwd_diff`, `bwd_diff`,
  `DifferentialPair`, `no_diff`, `[MaxIters]`, custom derivatives, and the list of
  things that cannot be differentiated.
  https://shader-slang.org/slang/user-guide/autodiff.html
- Autodiff tutorial (start here, it's short).
  https://docs.shader-slang.org/en/stable/auto-diff-tutorial-1.html
- SPIR-V target specifics — pointers, `PhysicalStorageBuffer64`, layout rules. This is
  the page for the §6.3 spike.
  https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/a2-01-spirv-target-specific.html
- Bangaru et al. *SLANG.D: Fast, Modular and Differentiable Shader Programming.*
  SIGGRAPH Asia 2023. The paper behind Slang's autodiff.

**Ours, for cross-reference**

- `app/viewer/shaders/util-raytrace.glsl:22` — `RayQueryResult`, already carries
  `(modelDrawIndex, primitiveIndex, barycentric)`: the walk's parameterization.
- `app/viewer/shaders/util-raytrace.glsl:44` — `utilTraceRayImpl`, the ray-query
  pattern to mirror in Slang.
- `app/viewer/shaders/util-raytrace.glsl:1632` — `utilIrradianceWalk`, the main path
  tracer; NEE sites around lines 575-830.
- `lib/ponder/shaders/util-mesh-raytrace.glsl:26` —
  `utilMeshAttributeDataFromIndices`, which already reconstructs interpolated
  position/normal/tangent from a hit triple. `dp/du` and `dn/du` fall out of the
  vertex fetch it already does.
- `lib/vkof/src/vkof.cpp:4082` — `compile_and_load_shader_module`, where `slangc`
  slots in beside `glslangValidator`.
