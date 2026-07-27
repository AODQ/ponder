#ifdef __cplusplus
#pragma once
#endif

#ifndef GLOBAL_PC_H
#define GLOBAL_PC_H

#ifdef __cplusplus
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>
#else
#ifndef f32
#define i32 int
#define i32v2 ivec2
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#define u32v2 uvec2
#define u32v3 uvec3
#define u64 uint64_t
#endif
#endif

#define VA(Type) u64

// per-frame values that are too large for the 128-byte root pushconstant
struct GpuGlobalExtended {
	f32 envIntensity;
	u32 renderWidth;
	u32 renderHeight;

	// matches ENV_MODE_* in environment.glsl
	i32 envMode;

	// subpixel jitter on the primary camera ray (antialiasing); off traces
	// every sample through the exact pixel center
	u32 antialias;

	// VIEW(0) / CLEAR(1) / PATHTRACE(2) -- see app/viewer/main.cpp's
	// FrameMode
	i32 frameMode;

	// single u32 atomic counter, zeroed by the cpu every frame; caps
	// util-nan-probe.glsl's debugPrintfEXT firings to
	// NAN_PROBE_LIMIT_PER_FRAME (see resolve.comp's nanProbeInit call)
	u64 nanProbeCounterVa;

	// material inspector: pixel coordinate to probe this frame, set by the
	// cpu on right-click. (-1,-1) means no probe requested. resolve.comp
	// writes the hit's modelDrawIndex/materialIndex into probeResult (a
	// plain host-readable buffer, not debug_printf -- doesn't need
	// validation layers enabled) when gl_GlobalInvocationID matches;
	// one-shot (the cpu resets this to (-1,-1) right after upload)
	i32v2 probePixel;
	// see probePixel above; the cpu resets both fields to -1 before
	// uploading a probe request, so it can tell a genuine hit from a stale
	// leftover value if the probed ray misses everything
	VA(GpuProbeResult) probeResult;

	// environment map, util-environment-map.glsl
	VA(EnvironmentMapHandles) envMap;
	// radians
	f32 envRotation;
	// next-event estimation toward the env map, mis-combined with the
	// existing bsdf-sampled technique; no-op (0) for every env mode other
	// than hdrmap regardless of this flag
	u32 envNeeEnabled;

	// diagnostic: terminate every path after its first surface hit (nee's
	// shadow ray still fires), so e.g. a grid of test spheres can't
	// cross-light itself -- only the primary ray + nee shadow rays touch
	// the AS
	u32 envMapOnlyMode;

	// caps a sample's luminance before it's written out (resolve.comp), so
	// the rare, very high-variance paths ordinary nee/mis can't reach --
	// e.g. a diffuse bounce that happens to hit a specular/glass object
	// which then refracts to a bright light, a caustic path nee can't
	// shortcut since it only helps at the vertex it's evaluated from --
	// don't blow out the image as fireflies. trades a small amount of
	// energy/bias for lower variance. <= 0 disables clamping entirely
	f32 fireflyClampLuminance;

	// scales the environment radiance a primary-ray miss writes to the
	// framebuffer (the visible backdrop, resolve.comp) -- indirect bounces
	// still see the full envIntensity, so this dims or brightens what the
	// camera sees without changing the lighting
	f32 envBackgroundIntensity;

	// vdb buffer
	VA(VdbHandle) vdb;
	// maps a world-space position/direction into the space the grid's own
	// world-to-index transform expects -- identity when the bound vdb has
	// no instance transform of its own (e.g. file view, or a stage vdb
	// instance left at the default transform). see stage::Transform /
	// stage::transform_to_inverse_m44; only the first active vdb instance
	// gets one, matching the single vdb slot above
	f32m44 vdbWorldToLocal;

	// TODO below should be per vdb-blob
	// rgb absorption/scattering coefficients (pbrt's HomogeneousMedium/
	// GridMedium sigma_a/sigma_s convention -- see lib/ponder/VDB_REFERENCES.md);
	// replaces a single scalar sigma + separate albedo vector so absorption
	// and scattering can each vary by channel independently. sigma_t =
	// vdbSigmaAbsorption + vdbSigmaScattering; the single-scattering albedo
	// utilVdbDeltaTrack's caller needs is vdbSigmaScattering/sigma_t,
	// derived at the collision rather than stored separately
	f32v3 vdbSigmaAbsorption;
	f32v3 vdbSigmaScattering;
	f32 vdbDropletDiameter;
	// manual Henyey-Greenstein anisotropy override (util-vdb.glsl's
	// utilVdbPhaseParamsForMedium), bypassing the droplet-diameter fog fit
	// above -- for non-water media (smoke soot, dust) the fit doesn't apply.
	// sentinel -2.0 (outside the valid [-1,1] g range) means "disabled, use
	// the droplet-diameter fit"; matches probePixel's (-1,-1)-means-off
	// convention above
	f32 vdbPhaseAnisotropyOverride;
	// blackbody emission from a second, optional "temperature" grid, sampled
	// at each real vdb collision alongside the density grid's scattering
	// event -- see lib/ponder/shaders/util-blackbody.glsl and
	// lib/ponder/VDB_REFERENCES.md. 0 means no temperature grid loaded (or
	// gridless -- same nanovdb float-grid sampling code as density, just
	// pointed at a different blob)
	VA(void) vdbTemperatureGrid;
	// raw grid values are almost never literal kelvin (e.g. houdini pyro's
	// "temperature" field is an unnormalized proxy, typically single or
	// double digits) -- utilBlackbodyRadiance's planck's-law evaluation is
	// exponentially sensitive to T, so a post-multiply on its output
	// (vdbEmissionScale below) can't compensate for the units being wrong
	// by orders of magnitude; this scales the sampled grid value into
	// kelvin *before* it hits planck's law
	f32 vdbTemperatureScale;
	// artist multiplier on utilBlackbodyRadiance's output -- see its own
	// header comment for why blackbody radiance isn't normalized to
	// absolute SI units
	f32 vdbEmissionScale;

	f32 fogThickness;
	f32v3 fogAlbedo;
	// fog's extent (world units); caps nee/free-flight distance
	f32 fogDistanceMax;
};

// root pushconstant, shared by every node in the frame's render graph;
// pinned to exactly vkof's 128-byte root pushconstant range so each node's
// own pushconstant struct starts at offset 128 in the shader-side PC block
struct GpuGlobalPc {
	f32 time;
	f32v3 cameraPos;
	f32 exposure;
	f32 pad0;
	f32m44 viewProj;
	VA(GpuGlobalExtended) extended;
	VA(GpuResolveModelIndirectBuffer) models;
	u64 pad1;
	u64 pad2;
	u64 pad3;
};

// what the material-inspector probe writes into (see
// GpuGlobalExtended.probeResult): plain host-readable buffer, not
// debug_printf, so it doesn't need validation layers enabled
struct GpuProbeResult {
	i32 modelDrawIndex;
	i32 materialIndex;
};

#ifdef __cplusplus
static_assert(sizeof(GpuGlobalPc) == 128, "GpuGlobalPC must be 128 bytes");
#else
layout(buffer_reference, scalar) buffer GpuGlobalExtendedBuffer {
	GpuGlobalExtended data;
};
layout(buffer_reference, scalar) buffer GpuProbeResultBuffer {
	GpuProbeResult data;
};
#endif

#endif // GLOBAL_PC_H
