import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-transmission";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 thetaSteps;
	u32 phiSteps;
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// full-sphere midpoint quadrature over wo for one transmission
// configuration per thread. two families of integrand, each accumulated
// both raw (the lib formula evaluated verbatim at every direction) and
// restricted to the sampler's exact image (the refract-image support test
// below): the raw-vs-restricted gap directly measures how much density /
// energy the formulas claim at wrong-branch (mirror-side) directions the
// production sampler can never emit.
//
// support test: reconstruct the walter half vector from (wi, wo), orient
// it to the normal's hemisphere exactly as the lib does, and accept wo
// only if refract()ing wi about that half vector actually lands back on
// wo. this covers every branch condition at once (correct refraction
// side, no total internal reflection, half vector in the sampled
// hemisphere).
//
// per config, 6 floats in: wi.xyz, ior, specularRoughness, isInsideMedium
// (0/1). output 8 floats: rawPdfIntegral, restrictedPdfIntegral,
// tirRestrictedReflectIntegral (vndf reflect density over the
// actually-TIR half-vector region -- the exact density of the sampler's
// tir fallback branch), tirLibReflectIntegral (openPbrTransmissionTirReflectPdf
// evaluated verbatim over the reflection side, the lib's deliberately
// unrestricted approximation), rawFCosIntegral, restrictedFCosIntegral
// (single-scatter transmittance; tint white, depth 0), unused, unused.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 6u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32 ior = inBuf[base+3u];
	const f32 roughness = inBuf[base+4u];
	const bool isInsideMedium = inBuf[base+5u] > 0.5f;

	OpenPbrMaterial mat;
	mat.specularIor = ior;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularRoughness = roughness;
	mat.geometryThinWalled = 0.0f;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 0.0f;

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	// frisvad basis, the frame the isotropic transmission path used internally
	const ShadingFrame frame = shadingFrameFromNormal(nor);
	const f32 etaI = isInsideMedium ? ior : 1.0f;
	const f32 etaT = isInsideMedium ? 1.0f : ior;

	const f32 dTheta = PI / f32(pc.thetaSteps);
	const f32 dPhi = TAU / f32(pc.phiSteps);

	f32 rawPdf = 0.0f;
	f32 restrictedPdf = 0.0f;
	f32 tirRestricted = 0.0f;
	f32 tirLib = 0.0f;
	f32 rawFCos = 0.0f;
	f32 restrictedFCos = 0.0f;

	for (u32 ti = 0u; ti < pc.thetaSteps; ++ti) {
		const f32 theta = (f32(ti) + 0.5f) * dTheta;
		const f32 cosTheta = cos(theta);
		const f32 sinTheta = sin(theta);
		for (u32 pj = 0u; pj < pc.phiSteps; ++pj) {
			const f32 phi = (f32(pj) + 0.5f) * dPhi;
			const f32v3 wo = (
				f32v3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta)
			);
			const f32 dOmega = sinTheta * dTheta * dPhi;

			const f32 pdf = (
				openPbrTransmissionPdf(mat, frame, wi, wo, isInsideMedium, ior)
			);
			const f32 f = (
				openPbrTransmissionEvaluateF(
					mat, frame, wi, wo, isInsideMedium, ior
				).x
			);
			rawPdf += pdf * dOmega;
			rawFCos += f * abs(wo.z) * dOmega;

			// refract-image support test
			const f32v3 ht = normalize(-(etaI * wi + etaT * wo));
			const f32v3 htOriented = dot(ht, nor) < 0.0f ? -ht : ht;
			const f32v3 refr = refract(-wi, htOriented, etaI / etaT);
			const bool reachable = (
				length(refr) > 0.5f
				&& dot(normalize(refr), wo) > 1.0f - 1e-5f
			);
			if (reachable) {
				restrictedPdf += pdf * dOmega;
				restrictedFCos += f * abs(wo.z) * dOmega;
			}

			// reflection-side density of the tir fallback branch:
			// openPbrTransmissionTirReflectPdf now restricts itself to
			// genuinely-TIR h's internally (fixed by the mixture-lobe-
			// selection milestone), so tirLib and tirRestricted are the
			// same quantity by construction -- kept as two separate
			// accumulators only to avoid changing this shader's output
			// stride / the c++ side's IntegrateResult layout
			{
				const f32 tirPdf = (
					openPbrTransmissionTirReflectPdf(
						mat, frame, wi, wo, isInsideMedium, ior
					)
				);
				tirLib += tirPdf * dOmega;
				tirRestricted += tirPdf * dOmega;
			}
		}
	}

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*8u+0u] = rawPdf;
	outBuf[i*8u+1u] = restrictedPdf;
	outBuf[i*8u+2u] = tirRestricted;
	outBuf[i*8u+3u] = tirLib;
	outBuf[i*8u+4u] = rawFCos;
	outBuf[i*8u+5u] = restrictedFCos;
	outBuf[i*8u+6u] = 0.0f;
	outBuf[i*8u+7u] = 0.0f;
}
