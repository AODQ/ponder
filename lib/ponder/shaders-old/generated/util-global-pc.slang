// autogen from global_pc.h by
// lib/tools/shared-header-to-slang.py

module "util-global-pc";
__exported import "util-types";

// (TODO REVIEW)
// GpuEmitterLightBuffer, for the emissive-mesh nee light list below
// (TODO REVIEW)

// per-frame values that are too large for the 128-byte root pushconstant
public struct GpuGlobalExtended {
	public f32 envIntensity;
	public u32 renderWidth;
	public u32 renderHeight;

	// matches ENV_MODE_* in environment.glsl
	public i32 envMode;

	// subpixel jitter on the primary camera ray (antialiasing); off traces
	// every sample through the exact pixel center
	public u32 antialias;

	// VIEW(0) / CLEAR(1) / PATHTRACE(2) -- see app/viewer/main.cpp's
	// FrameMode
	public i32 frameMode;

	// single u32 atomic counter, zeroed by the cpu every frame; caps
	// util-nan-probe.glsl's debugPrintfEXT firings to
	// NAN_PROBE_LIMIT_PER_FRAME (see resolve.comp's nanProbeInit call)
	public u64 nanProbeCounterVa;

	// material inspector: pixel coordinate to probe this frame, set by the
	// cpu on right-click. (-1,-1) means no probe requested. resolve.comp
	// writes the hit's modelDrawIndex/materialIndex into probeResult (a
	// plain host-readable buffer, not debug_printf -- doesn't need
	// validation layers enabled) when gl_GlobalInvocationID matches;
	// one-shot (the cpu resets this to (-1,-1) right after upload)
	public i32v2 probePixel;
	// see probePixel above; the cpu resets both fields to -1 before
	// uploading a probe request, so it can tell a genuine hit from a stale
	// leftover value if the probed ray misses everything
	public u64 probeResult; // VA(GpuProbeResult)

	// environment map, util-environment-map.glsl
	public u64 envMap; // VA(EnvironmentMapHandles)
	// radians
	public f32 envRotation;
	// next-event estimation toward the env map, mis-combined with the
	// existing bsdf-sampled technique; no-op (0) for every env mode other
	// than hdrmap regardless of this flag
	public u32 envNeeEnabled;

	// (TODO REVIEW)
	// emissive-mesh next-event estimation (lib/ponder's util-emitter.glsl).
	// the list is rebuilt every frame, so a material edit stays live; a stale
	// or empty list only costs variance, never correctness
	public u64 emitterLights; // VA(GpuEmitterLightBuffer)
	public u32 emitterLightCount;
	public u32 meshNeeEnabled;
	// (TODO REVIEW)
	// nee through a refracting caster, which a straight shadow ray reads as
	// blocked
	public u32 manifoldNeeEnabled;
	// (TODO REVIEW)
	// index of the environment map's entry in that same list, so one
	// power-weighted pick covers both techniques and one shadow ray serves
	// both. >= emitterLightCount means the env is not in the list, and the
	// env nee block falls back to standing on its own with pick pdf 1
	public u32 envLightIndex;
	// u32 counter array the emissive-nee instrumentation increments; only
	// read when util-raytrace.glsl is compiled with EMITTER_NEE_DEBUG, which
	// nothing but the unit tests does. 0 otherwise
	public u64 emitterNeeDebug; // VA(GpuEmitterNeeDebugBuffer)

	// (TODO REVIEW)
	// regir world-space light-selection grid (lib/ponder's util-regir.glsl)
	// slot claim words; 0 marks an empty slot
	public u64 regirGuards; // VA(GpuRegirGuardBuffer)
	// per-slot key, representative point/normal and normalization factors
	public u64 regirCells; // VA(GpuRegirCellBuffer)
	// skRegirReservoirCount entries per slot
	public u64 regirReservoirs; // VA(GpuRegirReservoirBuffer)
	// table capacity in slots, a power of two
	public u32 regirCellCount;
	// seeds the fill pass; must advance every frame
	public u32 regirFrameIndex;
	// world-space cell edge length
	public f32 regirCellSize;
	public u32 regirEnabled;
	// (TODO REVIEW)

	// (TODO REVIEW)
	// bounce index from which a bsdf-sampled continuation resolves straight
	// against the environment instead of tracing the AS; 0 disables it
	// entirely. 1 is the diagnostic (terminate after the first surface hit,
	// so e.g. a grid of test spheres can't cross-light itself); 2 is the
	// camera-drag preview, which keeps one real reflection/refraction
	// segment before falling back to the env. nee's shadow rays fire either
	// way
	public u32 envMapOnlyDepth;

	// caps a sample's luminance before it's written out (resolve.comp), so
	// the rare, very high-variance paths ordinary nee/mis can't reach --
	// e.g. a diffuse bounce that happens to hit a specular/glass object
	// which then refracts to a bright light, a caustic path nee can't
	// shortcut since it only helps at the vertex it's evaluated from --
	// don't blow out the image as fireflies. trades a small amount of
	// energy/bias for lower variance. <= 0 disables clamping entirely
	public f32 fireflyClampLuminance;

	// scales the environment radiance a primary-ray miss writes to the
	// framebuffer (the visible backdrop, resolve.comp) -- indirect bounces
	// still see the full envIntensity, so this dims or brightens what the
	// camera sees without changing the lighting
	public f32 envBackgroundIntensity;

	// vdb buffer
	public u64 vdb; // VA(VdbHandle)
	// maps a world-space position/direction into the space the grid's own
	// world-to-index transform expects -- identity when the bound vdb has
	// no instance transform of its own (e.g. file view, or a stage vdb
	// instance left at the default transform). see stage::Transform /
	// stage::transform_to_inverse_m44; only the first active vdb instance
	// gets one, matching the single vdb slot above
	public f32m44 vdbWorldToLocal;

	// TODO below should be per vdb-blob
	// rgb absorption/scattering coefficients (pbrt's HomogeneousMedium/
	// GridMedium sigma_a/sigma_s convention -- see lib/ponder/VDB_REFERENCES.md);
	// replaces a single scalar sigma + separate albedo vector so absorption
	// and scattering can each vary by channel independently. sigma_t =
	// vdbSigmaAbsorption + vdbSigmaScattering; the single-scattering albedo
	// utilVdbDeltaTrack's caller needs is vdbSigmaScattering/sigma_t,
	// derived at the collision rather than stored separately
	public f32v3 vdbSigmaAbsorption;
	public f32v3 vdbSigmaScattering;
	public f32 vdbDropletDiameter;
	// manual Henyey-Greenstein anisotropy override (util-vdb.glsl's
	// utilVdbPhaseParamsForMedium), bypassing the droplet-diameter fog fit
	// above -- for non-water media (smoke soot, dust) the fit doesn't apply.
	// sentinel -2.0 (outside the valid [-1,1] g range) means "disabled, use
	// the droplet-diameter fit"; matches probePixel's (-1,-1)-means-off
	// convention above
	public f32 vdbPhaseAnisotropyOverride;
	// blackbody emission from a second, optional "temperature" grid, sampled
	// at each real vdb collision alongside the density grid's scattering
	// event -- see lib/ponder/shaders/util-blackbody.glsl and
	// lib/ponder/VDB_REFERENCES.md. 0 means no temperature grid loaded (or
	// gridless -- same nanovdb f32-grid sampling code as density, just
	// pointed at a different blob)
	public u64 vdbTemperatureGrid; // VA(void)
	// raw grid values are almost never literal kelvin (e.g. houdini pyro's
	// "temperature" field is an unnormalized proxy, typically single or
	// double digits) -- utilBlackbodyRadiance's planck's-law evaluation is
	// exponentially sensitive to T, so a post-multiply on its output
	// (vdbEmissionScale below) can't compensate for the units being wrong
	// by orders of magnitude; this scales the sampled grid value into
	// kelvin *before* it hits planck's law
	public f32 vdbTemperatureScale;
	// artist multiplier on utilBlackbodyRadiance's output -- see its own
	// header comment for why blackbody radiance isn't normalized to
	// absolute SI units
	public f32 vdbEmissionScale;

	public f32 fogThickness;
	public f32v3 fogAlbedo;
	// fog's extent (world units); caps nee/free-flight distance
	public f32 fogDistanceMax;
};

// root pushconstant, shared by every node in the frame's render graph;
// pinned to exactly vkof's 128-byte root pushconstant range so each node's
// own pushconstant struct starts at offset 128 in the shader-side PC block
public struct GpuGlobalPc {
	public f32 time;
	public f32v3 cameraPos;
	public f32 exposure;
	public f32 pad0;
	public f32m44 viewProj;
	public u64 extended; // VA(GpuGlobalExtended)
	public u64 models; // VA(GpuResolveModelIndirectBuffer)
	public u64 pad1;
	public u64 pad2;
	public u64 pad3;
};

// what the material-inspector probe writes into (see
// GpuGlobalExtended.probeResult): plain host-readable buffer, not
// debug_printf, so it doesn't need validation layers enabled
public struct GpuProbeResult {
	public i32 modelDrawIndex;
	public i32 materialIndex;
};

// (TODO REVIEW)
// slots: 0 attempts, 1 sampler returned valid, 2 passed the f/cos gate,
// 3 shadow ray reported occluded, 4 contributed, 5 blocked by emissive
// geometry, 6 blocked by something else
public static const u32 EMITTER_NEE_DEBUG_SLOTS = 7u;

// (TODO REVIEW)
