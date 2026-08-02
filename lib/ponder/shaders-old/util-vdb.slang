// HAND-WRITTEN, keep in step with util-vdb.glsl. phase/fog/sigma are
// ported 1:1; every grid reader is a stub reporting an empty medium,
// since external-nanovdb.glsl is not ported.

module "util-vdb";
__exported import "util-types";
__exported import "util-random";
__exported import "util-shading-frame";
__exported import "util-fog";

// -----------------------------------------------------------------------------
// -- implementation detail
// -----------------------------------------------------------------------------

// route it through a buffer_reference so the grid blob is bound as a bda

#define PNANOVDB_GLSL

public struct UtilVdb {
	// always false: the grid reader is not ported
	public bool isValid;
	public u64 gridAddress;
	public f32 densityMax;
	// kept so callers that carry an instance transform still compile; the
	// stub never reads it
	public f32m44 worldToLocal;
};

// -----------------------------------------------------------------------------
// -- public api
// -----------------------------------------------------------------------------

// descriptor the cpu uploads next to the grid (ponder::vdb_load); the
// GpuGlobalExtended.vdb slot points here, not at the raw blob. layout must
// mirror GpuVdbHandle in lib/ponder/src/vdb.cpp (scalar layout)
public struct VdbHandle {
	public u64 vdbGrid;
	// grid-wide density maximum from the nanovdb stats; times the
	// integrator's sigma scale this is the delta/ratio-tracking majorant
	public f32 densityMax;
	public f32 pad0;
};

// bind the grid blob at `gridAddress`. Call once per invocation
// STUB: no grid reader in slang, so this reports an empty medium
public func utilVdbBind(
	gridAddress: u64,
) -> UtilVdb {
	UtilVdb vdb = {};
	vdb.gridAddress = gridAddress;
	return vdb;
}

// bind from the cpu-uploaded VdbHandle descriptor (what
// GpuGlobalExtended.vdb points at); picks up the grid address and the
// stats-derived densityMax in one go. worldToLocal is
// GpuGlobalExtended.vdbWorldToLocal -- the active vdb instance's inverse
// transform, identity if it has none
// STUB: no grid reader in slang, so this reports an empty medium
public func utilVdbBindHandle(
	handleAddress: u64,
	worldToLocal: f32m44,
) -> UtilVdb {
	UtilVdb vdb = {};
	vdb.gridAddress = handleAddress;
	return vdb;
}

// repoints the shared global bda buffer back at vdb's own grid; call this
// after binding/using any other grid (utilVdbBind) before touching vdb
// again -- see UtilVdb.gridAddress
// STUB: no grid reader in slang, so this reports an empty medium
public func utilVdbRebind(
	vdb: UtilVdb,
) {
}

// sample nearest-voxel value of a float grid at index-space position
// if outside of the grid's active topology, returns zero
// STUB: no grid reader in slang, so this reports an empty medium
public func utilVdbISDensity(
	inout vdb: UtilVdb,
	indexPos: f32v3,
) -> f32 {
	return 0.0f;
}

// sample nearest-voxel value of a float grid at world-space position
// if outside of the grid's active topology, returns zero
// TODO change to utilVdbWSDesnity
// STUB: no grid reader in slang, so this reports an empty medium
public func utilVdbDensity(
	inout vdb: UtilVdb,
	worldPos: f32v3,
) -> f32 {
	return 0.0f;
}

// index-space bbox of the grid's active voxels
// allows to clip rays before march
// STUB: no grid reader in slang, so this reports an empty medium
public func utilVdbIndexBbox(
	vdb: UtilVdb,
	out bboxMin: f32v3,
	out bboxMax: f32v3,
) {
	bboxMin = f32v3(0.0f);
	bboxMax = f32v3(0.0f);
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
public func utilVdbPhaseDraine(
	cosTheta: f32,
	alpha: f32,
	g: f32,
) -> f32 {
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

public struct UtilVdbPhaseParams {
	public f32 anisotropyHG;
	public f32 anisotropyDraine;
	public f32 draineAlpha;
	public f32 draineWeight;
};

// \phi_{fog}(\theta)
// fog phase function for anisotropic mediums
// mixture of Henyey-Greenstein and Draine phase functions
public func utilVdbPhaseFog(
	cosTheta: f32,
	p: UtilVdbPhaseParams,
) -> f32 {
	// \phi_{0,g_{HG}}, function reduces to Henyey-Greenstein when \alpha = 0
	const f32 hg = utilVdbPhaseDraine(cosTheta, 0.0f, p.anisotropyHG);

	// \phi_{\alpha, G_d}
	const f32 draine = (
		utilVdbPhaseDraine(cosTheta, p.draineAlpha, p.anisotropyDraine)
	);

	// \phi_{fog}(\theta) = \text{mix}(\phi_{0, g_{HG}}, \phi_{\alpha, G_d}, w_D)
	return lerp(hg, draine, p.draineWeight);
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
public func utilVdbPhaseFogDefaultParams(
	dropletDiameter: f32,
) -> UtilVdbPhaseParams {
	UtilVdbPhaseParams p;
	p.anisotropyHG = exp(-0.0990567f / (dropletDiameter - 1.67154f));
	p.anisotropyDraine = (
		exp(-2.20679f / (dropletDiameter + 3.91029f) - 0.428934f)
	);
	p.draineAlpha = exp(3.62489f - 8.29288f / (dropletDiameter + 5.52825f));
	p.draineWeight = exp(-0.599085f / (dropletDiameter - 0.641583f) - 0.665888f);
	return p;
}

// sentinel for "no manual anisotropy override, use the droplet-diameter fog
// fit instead" -- outside the valid [-1,1] Henyey-Greenstein g range, same
// off-sentinel convention as GpuGlobalExtended.probePixel's (-1,-1)
#define skUtilVdbPhaseAnisotropyOverrideOff (-2.0f)

// builds the phase-function params for a medium: either the physically-
// derived water-droplet fog fit (jendersie d'eon 2023), or -- when
// anisotropyOverride is inside [-1,1] -- a manual, artist-set plain
// Henyey-Greenstein lobe for media the droplet fit doesn't model (smoke
// soot, dust, other non-water participating media), matching how
// production tools (Houdini Pyro, Blender Principled Volume) expose
// anisotropy as a single direct g control rather than requiring a droplet
// size. draineWeight = 0 makes utilVdbPhaseFog/utilVdbPhaseSampleMixture
// reduce to plain HG (see utilVdbPhaseDraine's alpha=0 case)
public func utilVdbPhaseParamsForMedium(
	dropletDiameter: f32,
	anisotropyOverride: f32,
) -> UtilVdbPhaseParams {
	if (
		anisotropyOverride >= -1.0f && anisotropyOverride <= 1.0f
	) {
		UtilVdbPhaseParams p;
		p.anisotropyHG = anisotropyOverride;
		p.anisotropyDraine = 0.0f;
		p.draineAlpha = 0.0f;
		p.draineWeight = 0.0f;
		return p;
	}
	return utilVdbPhaseFogDefaultParams(dropletDiameter);
}

public func utilVdbPhaseSampleDraineCos(
	u: f32,
	gRaw: f32,
	a: f32,
) -> f32 {
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
	const f32 g2 = g * g;
	const f32 g3 = g * g2;
	const f32 g4 = g2 * g2;
	const f32 g6 = g2 * g4;
	const f32 pgp1_2 = (1 + g2) * (1 + g2);
	const f32 T1 = (-1 + g2) * (4 * g2 + a * pgp1_2);
	const f32 T1a = -a + a * g4;
	const f32 T1a3 = T1a * T1a * T1a;
	const f32 T2 = (
		-1296 * (-1 + g2) * (a - a * g2) * (T1a) * (4 * g2 + a * pgp1_2)
	);
	const f32 T3 = (
		3 * g2 * (1 + g * (-1 + 2 * u))
		+ a * (2 + g2 + g3 * (1 + 2 * g2) * (-1 + 2 * u))
	);
	const f32 T4a = 432 * T1a3 + T2 + 432 * (a - a * g2) * T3 * T3;
	const f32 T4b = -144 * a * g2 + 288 * a * g4 - 144 * a * g6;
	const f32 T4b3 = T4b * T4b * T4b;
	const f32 T4 = T4a + sqrt(-4 * T4b3 + T4a * T4a);
	const f32 T4p3 = pow(T4, 1.0 / 3.0);
	const f32 T6 = (
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
	const f32 T5 = 6 * (1 + g2) + T6;
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

public func utilVdbPhaseSampleHgCos(
	u: f32,
	g: f32,
) -> f32 {
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
public func utilVdbPhaseSampleMixture(
	p: UtilVdbPhaseParams,
	wi: f32v3,
	inout state: u64,
) -> f32v3 {
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

// per-channel (rgb) extinction coefficient: sigma_t = sigma_a + sigma_s
// (pbrt's HomogeneousMedium/GridMedium parameterization -- see
// VDB_REFERENCES.md); replaces the old scalar sigmaScale + separate albedo
// vector so absorption and scattering can each have their own spectral
// response (real colored media -- blood, milk, wax, dyed smoke -- absorb
// and scatter different wavelengths differently, which a single achromatic
// sigma times one albedo vector can't represent independently of the
// scattering fraction)
public func utilVdbSigmaT(
	sigmaAbsorption: f32v3,
	sigmaScattering: f32v3,
) -> f32v3 {
	return sigmaAbsorption + sigmaScattering;
}

// single-scattering albedo per channel: sigma_s / sigma_t. spatially
// constant even though sigma_t(x) itself varies with density(x) -- density
// is a common factor of both sigma_a(x) and sigma_s(x) here (only the
// material's sigmaAbsorption/sigmaScattering vectors are per-channel, the
// grid only ever scales them uniformly), so it cancels
public func utilVdbAlbedo(
	sigmaAbsorption: f32v3,
	sigmaScattering: f32v3,
) -> f32v3 {
	const f32v3 sigmaT = utilVdbSigmaT(sigmaAbsorption, sigmaScattering);
	return sigmaScattering / max(sigmaT, f32v3(1e-8f));
}

// unbiased transmittance estimate via free-flight delta-tracking; per-sample
// noise averages out in accumulation. useful for visibility functions.
// chromatic (rgb) ratio tracking (kutz et al. 2017, novák et al. 2018 --
// see VDB_REFERENCES.md): candidates are drawn against a single achromatic
// majorant, but the transmittance update is per-channel -- ratio tracking
// never branches, so this generalizes with no extra bookkeeping.
// STUB: no grid reader in slang, so this reports an empty medium
public func utilVdbRatioTrack(
	inout vdb: UtilVdb,
	worldOrigin: f32v3,
	worldDir: f32v3,
	distanceMax: f32,
	sigmaAbsorption: f32v3,
	sigmaScattering: f32v3,
	inout state: u64,
) -> f32v3 {
	return f32v3(1.0f);
}

// combined vdb ratio-tracked + fog beer-lambert transmittance toward a
// nee-sampled direction; each factor is gated independently. fogDistanceMax
// is GpuGlobalExtended.fogDistanceMax, the fog's imgui-editable extent
// (TODO REVIEW)
// travelDistance is how far the connection actually goes: 999999 for a shadow
// ray to the environment, the connection's own length for a finite one. a
// finite connection charged the full fogDistanceMax over-attenuates, and the
// bsdf technique traversing the same path does not -- so the two mis'd
// estimators disagree about a direction they must price identically
public func utilVdbFogTransmittance(
	inout vdb: UtilVdb,
	origin: f32v3,
	dir: f32v3,
	vdbSigmaAbsorption: f32v3,
	vdbSigmaScattering: f32v3,
	fogThickness: f32,
	fogDistanceMax: f32,
	travelDistance: f32,
	inout state: u64,
) -> f32v3 {
	f32v3 transmittance = f32v3(1.0f);
	if (vdb.isValid) {
		transmittance *= (
			utilVdbRatioTrack(
				vdb, origin, dir, travelDistance,
				vdbSigmaAbsorption, vdbSigmaScattering, state
			)
		);
	}
	if (fogThickness > 0.0f) {
		transmittance *= (
			utilFogTransmittance(
				fogThickness, min(travelDistance, fogDistanceMax)
			)
		);
	}
	return transmittance;
}
// (TODO REVIEW)

// free-flight delta-tracking through a vdb grid
// returns the distance to the next collision
// retunrs -1 if the ray does not meaningfully intersect the medium
//
// same as utilVdbRatioTrack, but returns the distance to the next collision
// rather than the transmittance probability
//
// chromatic (rgb) weighted delta tracking (kutz et al. 2017 -- see
// VDB_REFERENCES.md): unlike ratio tracking above, this function must make
// a hard real-vs-null decision at each candidate to report back a single
// collision distance, but the achromatic majorant walk can only use ONE
// (proxy) accept probability shared by all three channels. weighted delta
// tracking corrects for that by applying the true-channel/proxy-channel
// probability ratio at *every* step, real or null -- not just the last
// one, since the correction must compound consistently across however many
// null events precede the eventual real collision (an analog, unweighted
// walk -- correct only for a single/achromatic channel -- would otherwise
// silently bias the other two channels' apparent free-path distribution).
// outWeight carries that compounded per-channel correction back to the
// caller, to be folded into the collision's throughput alongside albedo
// STUB: no grid reader in slang, so this reports an empty medium
public func utilVdbDeltaTrack(
	inout vdb: UtilVdb,
	worldOrigin: f32v3,
	worldDir: f32v3,
	distanceMax: f32,
	sigmaAbsorption: f32v3,
	sigmaScattering: f32v3,
	inout state: u64,
	out outWeight: f32v3,
) -> f32 {
	outWeight = f32v3(1.0f);
	return distanceMax;
}

