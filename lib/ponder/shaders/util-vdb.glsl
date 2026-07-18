#ifndef UTIL_VDB_GLSL
#define UTIL_VDB_GLSL

#ifndef f32
#define i32v2 ivec2
#define i32 int
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#define u32v2 uvec2
#define u32v3 uvec3
#define u32v4 uvec4
#define u64 uint64_t
#endif

#include "util-random.glsl"
#include "util-shading-frame.glsl"

// -----------------------------------------------------------------------------
// -- implementation detail
// -----------------------------------------------------------------------------

// pnanovdb resolves against global pnanovdb_buf_data;
// route it through a buffer_reference so the grid blob is bound as a bda
layout(buffer_reference, scalar) buffer UtilVdbBdaBuffer { u32 data[]; };
UtilVdbBdaBuffer utilVdbBdaBuffer;
#define pnanovdb_buf_data utilVdbBdaBuffer.data

#define PNANOVDB_GLSL
#include "external-nanovdb.glsl"

// pnanovdb_buf_t carries no state in glsl mode, so pass constant sentinal
// to every pnanovdb call
const pnanovdb_buf_t skUtilVdbBuf = pnanovdb_buf_t(0u);

struct UtilVdb {
	// is false if scene has no vdb grid blob
	bool isValid;
	// grid blob handle
	pnanovdb_grid_handle_t grid;
	// caches the last-visited node path; for coherent lookups (marching a ray)
	pnanovdb_readaccessor_t accessor;
	// grid-wide density maximum (cpu-side nanovdb stats, via VdbHandle);
	// the delta/ratio-tracking majorant is this times the sigma scale
	f32 densityMax;
	// maps an incoming world-space position/direction into the space the
	// grid's own world-to-index transform expects (GpuGlobalExtended.
	// vdbWorldToLocal); identity when the caller has no instance transform
	// to apply. applied to positions with w=1 and directions with w=0
	// before every pnanovdb_grid_world_to_index[f/_dirf] call below
	f32m44 worldToLocal;
};

// -----------------------------------------------------------------------------
// -- public api
// -----------------------------------------------------------------------------

// descriptor the cpu uploads next to the grid (ponder::vdb_load); the
// GpuGlobalExtended.vdb slot points here, not at the raw blob. layout must
// mirror GpuVdbHandle in lib/ponder/src/vdb.cpp (scalar layout)
struct VdbHandle {
	u64 vdbGrid;
	// grid-wide density maximum from the nanovdb stats; times the
	// integrator's sigma scale this is the delta/ratio-tracking majorant
	f32 densityMax;
	f32 pad0;
};

layout(buffer_reference, scalar) buffer VdbHandleBuffer {
	VdbHandle data;
};


// bind the grid blob at `gridAddress`. Call once per invocation
UtilVdb utilVdbBind(const u64 gridAddress) {
	utilVdbBdaBuffer = UtilVdbBdaBuffer(gridAddress);
	UtilVdb vdb;
	vdb.grid.address = pnanovdb_address_null();
	vdb.densityMax = 0.0f;
	vdb.worldToLocal = f32m44(1.0f);
	pnanovdb_readaccessor_init(
		vdb.accessor,
		pnanovdb_tree_get_root(
			skUtilVdbBuf, pnanovdb_grid_get_tree(skUtilVdbBuf, vdb.grid)
		)
	);
	return vdb;
}

// bind from the cpu-uploaded VdbHandle descriptor (what
// GpuGlobalExtended.vdb points at); picks up the grid address and the
// stats-derived densityMax in one go. worldToLocal is
// GpuGlobalExtended.vdbWorldToLocal -- the active vdb instance's inverse
// transform, identity if it has none
UtilVdb utilVdbBindHandle(const u64 handleAddress, const f32m44 worldToLocal) {
	VdbHandle handle = VdbHandleBuffer(handleAddress).data;
	UtilVdb vdb = utilVdbBind(handle.vdbGrid);
	vdb.densityMax = handle.densityMax;
	vdb.worldToLocal = worldToLocal;
	return vdb;
}

// sample nearest-voxel value of a float grid at index-space position
// if outside of the grid's active topology, returns zero
f32 utilVdbISDensity(inout UtilVdb vdb, const f32v3 indexPos) {
	const pnanovdb_coord_t ijk = pnanovdb_hdda_pos_to_ijk(indexPos);
	return pnanovdb_read_float(
		skUtilVdbBuf,
		pnanovdb_readaccessor_get_value_address(
			PNANOVDB_GRID_TYPE_FLOAT, skUtilVdbBuf, vdb.accessor, ijk
		)
	);
}

// sample nearest-voxel value of a float grid at world-space position
// if outside of the grid's active topology, returns zero
// TODO change to utilVdbWSDesnity
f32 utilVdbDensity(inout UtilVdb vdb, const f32v3 worldPos) {
	const f32v3 localPos = (vdb.worldToLocal * f32v4(worldPos, 1.0f)).xyz;
	return (
		utilVdbISDensity(
			vdb,
			pnanovdb_grid_world_to_indexf(skUtilVdbBuf, vdb.grid, localPos)
		)
	);
}

// index-space bbox of the grid's active voxels
// allows to clip rays before march
void utilVdbIndexBbox(const UtilVdb vdb, out f32v3 bboxMin, out f32v3 bboxMax) {
	const pnanovdb_root_handle_t root = pnanovdb_tree_get_root(
		skUtilVdbBuf, pnanovdb_grid_get_tree(skUtilVdbBuf, vdb.grid)
	);
	const pnanovdb_coord_t lower = (
		pnanovdb_root_get_bbox_min(skUtilVdbBuf, root)
	);
	const pnanovdb_coord_t upper = (
		pnanovdb_root_get_bbox_max(skUtilVdbBuf, root)
	);
	bboxMin = f32v3(lower);
	// nanovdb bbox max is inclusive; +1 makes it an exclusive voxel bound
	bboxMax = f32v3(upper) + f32v3(1.0);
}

/*
	\begin{align*}
	&\phi_{\alpha, g}(\theta) =
		\frac{1}{4\pi}
		\frac{1 - g^2}{(1 + g^2 - 2g\cos\theta)^\frac{3}{2}}
		\frac{1 + \alpha \cos^2\theta}{1 + \frac{\alpha(1 + 2g^2)}{3}}
		\tag {jendersied'eon 2023, eq 2}
	\\
	&\phi_{fog}(\theta) =
		(1 - w_D) * \phi_{0, g_{HG}} + w_D * \phi_{\alpha, G_d}
		\tag{jendersie d'eon 2023, eq 3}
	\\
	&\text{where,}\\
	&g_{HG},\: G_D \text{are the anisotropy parameters}\\
	&\alpha \: \text{is the parameter of the Draine function}\\
	&w_D \: \text{is the mixture weight}
	\end{align*}
*/

// \phi_{\alpha, g}(\theta)
// draine phase function for anisotropic mediums
// reduces to,
// Henyey-Greenstein when \alpha = 0
// Rayleigh when g = 0, \alpha = 1
// cornette-shanks when \alpha = 1
f32 utilVdbPhaseDraine(
	const f32 cosTheta,
	const f32 alpha,
	const f32 g
) {
	// pdf is \frac{1}{4\pi}
	const f32 pdf = 1.0f/(4.0f * skPi);

	// HG term
	const f32 hgTerm = (
		// 1 - g^2
		(1.0f - g * g)
		// / (1 + g^2 - 2g\cos\theta)^\frac{3}{2}
		/ pow(1.0f + g * g - 2.0f * g * cosTheta, 1.5f)
	);

	// Draine term
	const f32 draineTerm = (
		// 1 + \alpha \cos^2\theta
		(1.0f + alpha * cosTheta * cosTheta)
		// / (1 + \frac{\alpha(1 + 2g^2)}{3})
		/ (1.0f + (alpha * (1.0f + 2.0f * g * g)) / 3.0f)
	);

	return pdf * hgTerm * draineTerm;
}

struct UtilVdbPhaseParams {
	f32 anisotropyHG;
	f32 anisotropyDraine;
	f32 draineAlpha;
	f32 draineWeight;
};

// \phi_{fog}(\theta)
// fog phase function for anisotropic mediums
// mixture of Henyey-Greenstein and Draine phase functions
f32 utilVdbPhaseFog(
	const f32 cosTheta,
	const UtilVdbPhaseParams p
) {
	// \phi_{0,g_{HG}}, function reduces to Henyey-Greenstein when \alpha = 0
	const f32 hg = utilVdbPhaseDraine(cosTheta, 0.0f, p.anisotropyHG);

	// \phi_{\alpha, G_d}
	const f32 draine = (
		utilVdbPhaseDraine(cosTheta, p.draineAlpha, p.anisotropyDraine)
	);

	// \phi_{fog}(\theta) = \text{mix}(\phi_{0, g_{HG}}, \phi_{\alpha, G_d}, w_D)
	return mix(hg, draine, p.draineWeight);
}

/*
	&g_{HG}(d) = e^{-\frac{0.0990567}{d-1.67154}}
		\tag{jendersied'eon 2023, eq 4}
	\\
	&g_D(d) = e^{-\frac{2.20679}{d+3.91029} - 0.428934}
		\tag{jendersied'eon 2023, eq 5}
	\\
	&\alpha(d) = e^{3.62489 - \frac{8.29288}{d+5.52825}}
		\tag{jendersied'eon 2023, eq 6}
	\\
	&w_D(d) = e^{-\frac{0.599085}{d-0.641583} - 0.665888}
		\tag{jendersied'eon 2023, eq 7}
	\\
*/
// reasonable default parameters for the fog phase function,
// fitted over a range of water droplet diameter 5 < d < 50 \mu m
UtilVdbPhaseParams utilVdbPhaseFogDefaultParams(const f32 dropletDiameter) {
	UtilVdbPhaseParams p;
	p.anisotropyHG = exp(-0.0990567f / (dropletDiameter - 1.67154f));
	p.anisotropyDraine = (
		exp(-2.20679f / (dropletDiameter + 3.91029f) - 0.428934f)
	);
	p.draineAlpha = exp(3.62489f - 8.29288f / (dropletDiameter + 5.52825f));
	p.draineWeight = exp(-0.599085f / (dropletDiameter - 0.641583f) - 0.665888f);
	return p;
}

f32 utilVdbPhaseSampleDraineCos(
	const f32 u,
	const f32 gRaw,
	const f32 a
) {
	// mirrors utilVdbPhaseSampleHgCos's isotropic guard: the closed-form
	// cubic solve below divides by 2g at the end, which amplifies float32
	// rounding noise picked up earlier in the solve as g -> 0; measured
	// unsafe (|cosTheta| overshoots [-1, 1] by more than 1e-3) up to
	// |g| <= 0.042 at alpha <= 40. floor g away from zero rather than
	// special-case exactly 0, since the draine lobe has no closed form of
	// its own at g=0 (unlike hg's 1-2u)
	const f32 g = (
		abs(gRaw) < 0.05f
			? (gRaw >= 0.0f ? 0.05f : -0.05f)
			: gRaw
	);
	// formula not shown, from Jendersie d'eon 2023
	const float g2 = g * g;
	const float g3 = g * g2;
	const float g4 = g2 * g2;
	const float g6 = g2 * g4;
	const float pgp1_2 = (1 + g2) * (1 + g2);
	const float T1 = (-1 + g2) * (4 * g2 + a * pgp1_2);
	const float T1a = -a + a * g4;
	const float T1a3 = T1a * T1a * T1a;
	const float T2 = (
		-1296 * (-1 + g2) * (a - a * g2) * (T1a) * (4 * g2 + a * pgp1_2)
	);
	const float T3 = (
		3 * g2 * (1 + g * (-1 + 2 * u))
		+ a * (2 + g2 + g3 * (1 + 2 * g2) * (-1 + 2 * u))
	);
	const float T4a = 432 * T1a3 + T2 + 432 * (a - a * g2) * T3 * T3;
	const float T4b = -144 * a * g2 + 288 * a * g4 - 144 * a * g6;
	const float T4b3 = T4b * T4b * T4b;
	const float T4 = T4a + sqrt(-4 * T4b3 + T4a * T4a);
	const float T4p3 = pow(T4, 1.0 / 3.0);
	const float T6 = (
		(
			2 * T1a
			+ (
				(48 * pow(2, 1.0 / 3.0) * (-(a * g2) + 2 * a * g4 - a * g6))
				/ T4p3
			)
			+ T4p3 / (3. * pow(2, 1.0 / 3.0))
		)
		/ (a - a * g2)
	);
	const float T5 = 6 * (1 + g2) + T6;
	return (
		(
			1 + g2
			- pow(
				(
					-0.5 * sqrt(T5)
					+ sqrt(
						6 * (1 + g2) - (8 * T3) / (a * (-1 + g2) * sqrt(T5)) - T6
					)
					/ 2.
				),
				2
			)
		)
		/ (2. * g)
	);
}

f32 utilVdbPhaseSampleHgCos(
	const f32 u,
	const f32 g
) {
	// isotropic limit
	if (abs(g) < 1e-3f) {
		return 1.0f - 2.0f * u;
	}
	// standard HG inversion
	const f32 s = (1 - g * g) / (1.0f - g + 2.0f * g * u);
	return (1.0f + g * g - s * s) / (2.0f * g);
}

// Draine CDF inversion, math not here but made available
// https://research.nvidia.com/labs/rtr/approximate-mie/assets/draine.hlsl
f32v3 utilVdbPhaseSampleMixture(
	const UtilVdbPhaseParams p,
	const f32v3 wi,
	inout u64 state
) {
	// sample the mixture
	const f32 cosTheta = (
		fnSampleUniform(state) < p.draineWeight
		? (
			utilVdbPhaseSampleDraineCos(
				fnSampleUniform(state),
				p.anisotropyDraine,
				p.draineAlpha
			)
		)
		: utilVdbPhaseSampleHgCos(fnSampleUniform(state), p.anisotropyHG)
	);
	const f32 phi = skTau * fnSampleUniform(state);
	// frisvald basis around \omega_i, then spherical-to-cartesian
	const ShadingFrame frame = shadingFrameFromNormal(wi);
	const f32 sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
	return (
		frame.tanX * (sinTheta*cos(phi))
		+ frame.tanY * (sinTheta*sin(phi))
		+ frame.nor * cosTheta
	);
}

// free-flight delta-tracking through a vdb grid
// unbiased estimate of the transmittance along the segment
// returned value is the random estimate of the transmittance,
// per-sample noise averages the out in accumulation.
// useful for visibility functions
f32 utilVdbRatioTrack(
	inout UtilVdb vdb,
	const f32v3 worldOrigin,
	const f32v3 worldDir,
	const f32 distanceMax,
	const f32 sigmaScale,
	inout u64 state
) {
	const f32 sigmaMajorant = vdb.densityMax * sigmaScale;
	if (sigmaMajorant <= 0.0f) {
		return 1.0f;
	}

	// -- clip to active voxel bbox
	f32v3 bboxMin;
	f32v3 bboxMax;
	utilVdbIndexBbox(vdb, bboxMin, bboxMax);
	const f32v3 localOrigin = (
		(vdb.worldToLocal * f32v4(worldOrigin, 1.0f)).xyz
	);
	const f32v3 localDir = (vdb.worldToLocal * f32v4(worldDir, 0.0f)).xyz;
	const f32v3 vdbOrigin = (
		pnanovdb_grid_world_to_indexf(skUtilVdbBuf, vdb.grid, localOrigin)
	);
	const f32v3 vdbDir = (
		pnanovdb_grid_world_to_index_dirf(skUtilVdbBuf, vdb.grid, localDir)
	);

	// -- compute the bounding box intersection
	const f32v3 tMin = (bboxMin - vdbOrigin) / vdbDir;
	const f32v3 tMax = (bboxMax - vdbOrigin) / vdbDir;
	const f32 t0 = (
		max(
			max(min(tMin.x, tMax.x), min(tMin.y, tMax.y)),
			max(min(tMin.z, tMax.z), 0.0f)
		)
	);
	const f32 t1 = (
		min(
			min(max(tMin.x, tMax.x), max(tMin.y, tMax.y)),
			min(max(tMin.z, tMax.z), distanceMax)
		)
	);
	if (t1 <= t0) {
		return 1.0f;
	}

	// -- free-flight sampling, collecting the probability weight
	f32 transmittance = 1.0f;
	f32 t = t0;
	for (i32 i = 0; i < 1000; ++i) {
		t -= log(1.0f - fnSampleUniform(state)) / sigmaMajorant;
		if (t >= t1) {
			break;
		}
		const f32 density = utilVdbISDensity(vdb, vdbOrigin + vdbDir * t);
		transmittance *= (1.0f - density / vdb.densityMax);
		// if the transmittance is low, fails the visibility test and can early
		// exit
		if (transmittance < 1e-6f) {
			return 0.0f;
		}
	}
	return transmittance;
}

// -----------------------------------------------------------------------------
// -- homogeneous fog
// -----------------------------------------------------------------------------

#define skUtilFogPhaseIsotropic (1.0f / (4.0f * skPi))

// sigma is extinction per world unit (GpuGlobalExtended.fogThickness);
// caller guards sigma > 0. free-flight distance through constant sigma is
// exactly exponential and transmittance is closed-form beer-lambert -- no
// majorant/rejection loop needed, unlike the heterogeneous vdb below
f32 utilFogSampleDistance(const f32 sigma, inout u64 state) {
	return -log(1.0f - fnSampleUniform(state)) / sigma;
}

f32 utilFogTransmittance(const f32 sigma, const f32 dist) {
	return exp(-sigma * dist);
}

// combined vdb ratio-tracked + fog beer-lambert transmittance toward a
// nee-sampled direction; each factor is gated independently. fogDistanceMax
// is GpuGlobalExtended.fogDistanceMax, the fog's imgui-editable extent
f32 utilVdbFogTransmittance(
	inout UtilVdb vdb,
	const f32v3 origin,
	const f32v3 dir,
	const f32 vdbSigmaScale,
	const f32 fogThickness,
	const f32 fogDistanceMax,
	inout u64 state
) {
	f32 transmittance = 1.0f;
	if (vdb.isValid) {
		transmittance *= (
			utilVdbRatioTrack(vdb, origin, dir, 999999.0f, vdbSigmaScale, state)
		);
	}
	if (fogThickness > 0.0f) {
		transmittance *= utilFogTransmittance(fogThickness, fogDistanceMax);
	}
	return transmittance;
}

// free-flight delta-tracking through a vdb grid
// returns the distance to the next collision
// retunrs -1 if the ray does not meaningfully intersect the medium
//
// same as utilVdbRatioTrack, but returns the distance to the next collision
// rather than the transmittance probability
f32 utilVdbDeltaTrack(
	inout UtilVdb vdb,
	const f32v3 worldOrigin,
	const f32v3 worldDir,
	const f32 distanceMax,
	const f32 sigmaScale,
	inout u64 state
) {
	const f32 sigmaMajorant = vdb.densityMax * sigmaScale;
	if (sigmaMajorant <= 0.0f) {
		return -1.0f;
	}

	// -- clip to active voxel bbox
	f32v3 bboxMin;
	f32v3 bboxMax;
	utilVdbIndexBbox(vdb, bboxMin, bboxMax);
	const f32v3 localOrigin = (
		(vdb.worldToLocal * f32v4(worldOrigin, 1.0f)).xyz
	);
	const f32v3 localDir = (vdb.worldToLocal * f32v4(worldDir, 0.0f)).xyz;
	const f32v3 vdbOrigin = (
		pnanovdb_grid_world_to_indexf(skUtilVdbBuf, vdb.grid, localOrigin)
	);
	const f32v3 vdbDir = (
		pnanovdb_grid_world_to_index_dirf(skUtilVdbBuf, vdb.grid, localDir)
	);

	// -- compute the bounding box intersection
	const f32v3 tMin = (bboxMin - vdbOrigin) / vdbDir;
	const f32v3 tMax = (bboxMax - vdbOrigin) / vdbDir;
	const f32 t0 = (
		max(
			max(min(tMin.x, tMax.x), min(tMin.y, tMax.y)),
			max(min(tMin.z, tMax.z), 0.0f)
		)
	);
	const f32 t1 = (
		min(
			min(max(tMin.x, tMax.x), max(tMin.y, tMax.y)),
			min(max(tMin.z, tMax.z), distanceMax)
		)
	);
	if (t1 <= t0) {
		return -1.0f;
	}

	// -- free-flight sampling
	// exponential steps against majorant.
	// Collision becomes probability sigmaT(x)/sigmaMajorant,
	// where sigmaT(x) is the density at the current sample position.
	// the sigmaScale cancels out out, leaving density/densityMax.
	// Null collision carries no weight and re-samples from new position.
	// iteration cap is to avoid infinite loops, expected sigmaMajorant*(t1-t0)
	f32 t = t0;
	for (i32 i = 0; i < 1000; ++i) {
		t -= log(1.0f - fnSampleUniform(state)) / sigmaMajorant;
		if (t >= t1) {
			break;
		}
		const f32 density = utilVdbISDensity(vdb, vdbOrigin + vdbDir * t);
		if (fnSampleUniform(state) * vdb.densityMax < density) {
			return t;
		}
	}
	return -1.0f;
}

#endif // UTIL_VDB_GLSL
