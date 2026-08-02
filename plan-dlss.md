# DLSS Ray Reconstruction — opt-in "what would DLSS do" mode

Written 2026-11-21. An optional viewer mode: every frame is 1 spp and goes through DLSS-RR
instead of accumulation and the à-trous filter. It is off by default, both at build time and at
runtime. Progressive accumulation (the reference path) stays as it is.

Source: `doc/DLSS-RR Integration Guide.pdf` in github.com/NVIDIA/DLSS (the August 2026 revision).

## Facts checked

- Linux and Vulkan are supported. `lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.310.9.16` is the RR
  library, and it links against `libnvsdk_ngx.a`. Create the feature with
  `NVSDK_NGX_VULKAN_CreateFeature1(..., NVSDK_NGX_Feature_SuperSamplingDenoising, ...)`
  through the `NGX_VULKAN_CREATE_DLSSD_EXT1` helper.
- Requirements are any RTX GPU and driver ≥ 580. This machine has a 2080 Ti and driver 610.57, so both are fine.
- License: "NVIDIA RTX SDKs LICENSE". It is proprietary, so the SDK stays out of the repo and
  is fetched only when the build option is on.
- RR assumes samples are **independent, white-noise** samples. Ponder's path sampler is
  pcg3d-hashed (`util/noise.slang`), which suits it. Nothing blue-noise may feed the path
  sampler in this mode.
- Exposure, auto-exposure and sharpening are ignored by RR. Tonemapping happens after RR, on its output.
- `InReset` is set on scene transitions: model or stage load, and camera teleports.

## Inputs RR needs vs what exists

| input (NGX name) | format | ponder today |
|---|---|---|
| `pInColor`, noisy linear radiance | RGB16F | the 1 spp output before accumulation exists |
| `pInDiffuseAlbedo` | RGB | the g-buffer `albedo` is close, but has to be split into diffuse and specular |
| `pInSpecularAlbedo` | RGB | missing; the guide's appendix has an approximation, or the energy tables can be used |
| `pInNormals` + `pInRoughness` | RGB16F world, R | the g-buffer `nor` exists; roughness is missing |
| `pInDepth` | R32F | derivable from the g-buffer `pos` |
| `pInMotionVectors` | RG16F, render resolution | missing; camera-only motion from `pos` and the previous view-projection |
| `pInSpecularHitDistance` | R16F | optional and missing; this is for later (better reflections in motion) |
| `pInOutput` | RGBA16F, output resolution | a new image feeding resolve/bloom |
| jitter offset | per frame | **mismatch.** `wavefront-camera.slang` jitters each sample randomly per pixel, but RR wants one global subpixel offset per frame (Halton, at least 32 phases), passed as `JitterOffsetX/Y` |

## Steps

Each step is small and verifiable on its own. Steps 0–2 change nothing visible.

0. **Build plumbing.** Add a CMake option `PONDER_DLSS`, defaulting to OFF. When it's on, fetch the
   DLSS SDK into `build/_deps`, link `libnvsdk_ngx.a`, copy `libnvidia-ngx-dlssd.so*` next to `viewer`,
   and define `PONDER_DLSS=1`. With the option OFF, the build is unchanged.
1. **vkof: NGX Vulkan extensions.** Query `NVSDK_NGX_VULKAN_RequiredExtensions` and append the
   results to the instance and device extensions in vk-bootstrap, only when `PONDER_DLSS` is on. Call
   `NVSDK_NGX_VULKAN_Init` after device creation and log whether RR is available.
2. **Guide AOVs.** Write depth, motion vectors, roughness and specular albedo at the primary hit,
   only when RR mode is on. Motion vectors come from the camera only, using the previous frame's
   view-projection. *This is slang AOV work, so it's yours by default. Say whether you want it
   treated as plumbing.*
3. **Per-frame global jitter.** In RR mode, a Halton(2,3) offset with ≥32 phases replaces the
   per-sample `PixelJitter` and is passed to RR.
4. **Evaluate.** Create the feature lazily on first enable, at the render resolution with the
   output at the same size (DLAA, no upscaling). Each frame, run path trace → RR eval → resolve.
   Skip `pt-accumulate` and `pt-denoise`. Set `InReset` when anything other than camera motion
   would reset accumulation.
5. **UI.** Add a "DLSS-RR (realtime)" checkbox in the viewer, unchecked at startup. Greyed out with a reason
   when the build lacks DLSS or the feature isn't available. Also add a render-preset combo, and an
   "A/B vs accumulation" hint saying that stopping the camera does *not* switch back to accumulation
   while the mode is on.
6. **Later, optional.** Upscaling modes (render below the output resolution), specular hit distance,
   and glass/transparency guide blending (the guide's §3.4 notes on refracting surfaces).

## Validation

- With the mode off, images must be byte-identical to before. The existing screenshot path with `--spp` is the check.
- With the mode on, check a static camera on DamagedHelmet (it should converge to look like
  accumulation, apart from RR's bias), a camera orbit on Sponza (no shimmer), and glass scenes
  (CompareTransmission, DragonDispersion) to see how RR handles refraction without specular
  motion vectors.
