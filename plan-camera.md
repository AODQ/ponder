# Camera model

Written 2026-11-22. The camera has a thin lens; this plan covers what a
spectral renderer with a gr transport can add on top. Items are ordered by
cost and by how much they depend on each other, not by importance.

---

## 0. Status

Done (2026-11-22):
- `Camera_ThinLens` in `util/camera.slang`, wrapping `Camera_Pinhole`; lens
  radius 0 returns the pinhole ray. Concentric disk sample, focal plane
  perpendicular to forward.
- `SampleDimension.CameraLens` (2 wide) before `BounceStart`. This shifted
  every bounce dimension by 2: renders converge to the same image, noise is
  not byte-identical with older captures.
- globals `cameraLensRadius`, `cameraFocalDistance` (GpuGlobal 544 / 548).
- viewer: lens radius (log slider, 0 to 0.1 m), focal distance, `b+click`
  focuses on the first hit under the cursor (through the debug path
  capture), `b+wheel` steps the lens radius geometrically.
- debug path dimension names include `CameraLensX/Y`.

Known gaps in what is done:
- the debug path reuses the view it was picked from but the *current* lens
  settings; changing the aperture after picking replays a different ray.
- `CameraSample.lens` comment says "unit disk [-1, 1]^2"; it is the
  [0, 1)^2 square sample.
- regir-view jitters the lens too; check that is wanted for a debug view.

---

## 1. Constraint that shapes the spectral items: hero wavelengths

A path carries 4 wavelengths that share one geometric path. Anything at the
camera that makes the *ray* depend on wavelength (dispersion in the lens,
wavelength-dependent magnification) breaks that sharing. Two ways out:

- **collapse at the camera**: keep the hero wavelength, drop the other three
  (as dispersive glass does). Correct, but costs ~4x colour noise on every
  pixel, not only on glass.
- **per-wavelength pdf (spectral MIS)**: trace the hero's ray, weight the
  secondaries by the ratio of their camera pdf to the hero's. Only works if
  each wavelength's camera ray distribution covers the others'; a thin lens
  with different focus per wavelength does overlap (same lens disk, different
  directions), so this is plausible but needs a derivation before code.

Everything in §3 is free of this; everything in §4 runs into it.

---

## 2. Cheap, not spectral

Each is an evening or less, and independent.

### 2.1 Physical lens parameters
Replace "lens radius in meters" with photographic controls:
- sensor size (default 36 x 24 mm full frame)
- focal length f from fov: `f = (sensorHeight / 2) / tan(fovY / 2)`
- f-number N: `lensRadius = f / (2 N)`

UI: f-number (log, f/1 to f/32) plus focal length in mm, fov derived; keep
the radius as a readout. Scene must be in meters for the numbers to mean
anything (glTF is). `b+wheel` steps in third stops.

### 2.2 Aperture blades (polygonal bokeh)
Sample an n-gon instead of the disk: pick a triangle uniformly, sample it
uniformly. Blade count and rotation as globals. Curved blades: blend the
polygon radius toward the circle. Next step: an aperture *image* (alias
table over its pixels) for arbitrary bokeh shapes.

### 2.3 Rolling shutter
Shutter time per row instead of per pixel: `time = open + (row / height +
jitter) * readout + shutter * exposure`. One extra global (readout duration).
Shows well on the rotating accretion disk.

### 2.4 Tilt-shift
- shift: offset the image plane window (add to ndc before building the
  direction). Pure projection change.
- tilt: focal *plane* not perpendicular to forward (Scheimpflug). Replace the
  `focalDistance / dot(d, forward)` intersection with a ray-plane hit against
  a plane with its own normal. Two angles as globals.

### 2.5 Vignetting
- natural: cos^4 of the angle off axis as a camera ray weight. Decide whether
  it belongs in the camera weight or as a sensor post-process; most renderers
  leave it out by default.
- mechanical / cat-eye: clip the lens sample against a second, offset disk
  (the barrel) whose offset grows with the image-plane radius. Gives the
  lemon-shaped bokeh near the frame edges.

---

## 3. Spectral, no transport change

### 3.1 Sensor spectral response
Today the spectrum goes to colour through the CIE observer (check where:
the accumulate/output path). Add a camera mode:
- per-channel spectral sensitivity curves (r, g, b) as tables, from a
  published camera SSF dataset *(choose and check licence before shipping
  data)*
- a 3x3 matrix from camera rgb to the working space, fitted on a reflectance
  chart under the scene white
- presets: CIE observer (current), a digital camera, an infrared-converted
  sensor (sensitivity extended past 700 nm; needs the wavelength range to
  extend, check `wavelengthSampleUniform` bounds)

The sampled wavelength range must cover the sensor's support; an IR preset
widens it, which raises variance for the visible range unless sampling is
importance-weighted by the summed sensitivity.

### 3.2 Filters
A spectral transmittance multiplied into the camera weight: UV/IR cut,
coloured gels, ND. Trivial once 3.1 has a spectral table path.

### 3.3 Diffraction starbursts
Post-process: the far-field pattern of the aperture (|FFT|^2 of the aperture
mask) scales linearly with wavelength. Convolve bright pixels with the
pattern summed over wavelengths. Reuses the blade shape from 2.2. Lives in
the output stage, not the path tracer; needs a spectral (or at least
multi-band) image, so do it from per-band accumulation or approximate with
the rgb image and a per-channel scale.

---

## 4. Spectral, breaks hero sharing (see §1)

### 4.1 Longitudinal chromatic aberration
Thin lens with `focalDistance(λ)` from a glass dispersion model
(Sellmeier coefficients for e.g. N-BK7). Lens maker's equation gives the
focal length per λ; with a fixed sensor distance, the in-focus distance
moves with λ. Start with collapse-at-camera to get a reference image, then
try the spectral MIS weighting from §1 against it.

### 4.2 Lateral chromatic aberration
Magnification as a function of λ (scale ndc by `1 + k (λ - λref)`). Same
hero problem, same approach.

### 4.3 Realistic lens systems
Trace through a lens prescription (radius, thickness, glass, aperture per
element) with Sellmeier glass, as pbrt-v4's RealisticCamera does. Produces
real spherical aberration, coma, field curvature and chromatic aberration
from first principles. Needs: exit pupil precomputation for efficient
sampling, focus by moving the element stack. Large; only after 4.1 settles
how wavelength-dependent camera rays are handled.

---

## 5. Relativistic observer

The most distinctive item for this renderer: spectral + gr already exist.

An observer moving at velocity v sees:
- **aberration**: directions bunch toward the motion
  (`cos θ' = (cos θ + β) / (1 + β cos θ)`)
- **Doppler shift**: frequency scales by `D = 1 / (γ (1 - β cos θ))`
- **searchlight**: radiance scales by `D^4` for broadband (specific intensity
  `I_ν / ν^3` is invariant, so `I_ν' = D^3 I_ν` per frequency; spectral
  rendering does the right thing if it samples λ in the observer frame and
  shifts to the emitter frame)

Plan:
1. **flat space, constant velocity.** At ray generation: build the ray in the
   observer frame, Lorentz-transform the direction to the static frame,
   shift the sampled wavelengths by D, carry D^3 as a weight. The path's
   wavelength shift should reuse whatever the transport already does for
   gravitational redshift (debug vertices carry `frequencyShift`; check how
   it is applied before writing new code). Hero sharing survives: all four
   wavelengths shift by the same factor along one direction.
2. **near the hole.** The observer's velocity is relative to the local static
   frame (or a zamo frame for kerr). Build the local tetrad at the camera
   position, boost it, and start the geodesic from the boosted tetrad. An
   orbiting observer then sees the correct combined gravitational + kinematic
   shift.
3. **wavelength range.** Strong shifts move visible light out of range and
   pull in uv/ir from the emitters. The sampled observer-frame range stays
   visible; emitter spectra must be defined over the shifted range (blackbody
   is; rgb-uplifted textures are not, they need a policy for out-of-range).

---

## 6. Order

1. 2.1 physical lens parameters (makes the existing thin lens usable)
2. 3.1 sensor response (cheap, clearly spectral, unlocks 3.2 and 3.3)
3. 5.1 relativistic observer, flat space
4. 2.2 blades, 2.3 rolling shutter, 2.4 tilt-shift, 2.5 vignetting as filler
5. 4.1 chromatic aberration with collapse, then the MIS variant
6. 5.2 relativistic observer near the hole
7. 3.3 starbursts, 4.2, 4.3 when the rest has settled

Not in roadmap yet: slot this into `plan-roadmap.md` once the caustic solver
(step 7) is underway; none of it blocks or is blocked by that work.
