import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-metallic";
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

// furnace test with thin-film interference active: same fixed-wi hemisphere
// integral as metallic_lobe_furnace_integrate.comp, but the fresnel term is
// now the thin-film interference color instead of plain f82 -- unlike f82
// (a function of dot(nor, wi) only), the thin-film term is evaluated at
// dot(h, wi), which varies across the wo integration domain, so it does
// NOT factor out of the integral the way the plain-metal furnace derivation
// relies on. output is per-channel (rgb), not a scalar: iridescence
// redistributes energy across wavelengths rather than preserving a flat
// spectrum, so a colored result here is expected, not a bug -- the
// invariant under test is that the redistribution stays energy-bounded
// (does not manufacture energy), not that it stays achromatic.
//
// per sample, 10 floats: muI, specularRoughness, specularRoughnessAnisotropy,
// baseWeight, baseColorScalar, specularWeight, specularColorScalar,
// thinFilmWeight, thinFilmThickness, thinFilmIor

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 10u;
	const f32 muI = inBuf[base+0u];

	OpenPbrMaterial mat;
	mat.specularRoughness = inBuf[base+1u];
	mat.specularRoughnessAnisotropy = inBuf[base+2u];
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.baseWeight = inBuf[base+3u];
	mat.baseColor = f32v3(inBuf[base+4u]);
	mat.specularWeight = inBuf[base+5u];
	mat.specularColor = f32v3(inBuf[base+6u]);
	mat.coatWeight = 0.0f;
	mat.coatRoughness = 0.0f;
	mat.thinFilmWeight = inBuf[base+7u];
	mat.thinFilmThickness = inBuf[base+8u];
	mat.thinFilmIor = inBuf[base+9u];

	ShadingFrame frame;
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	frame.nor = f32v3(0.0f, 0.0f, 1.0f);

	const f32v3 wi = f32v3(sqrt(max(0.0f, 1.0f - muI * muI)), 0.0f, muI);

	const f32 dTheta = (PI * 0.5f) / f32(pc.thetaSteps);
	const f32 dPhi = (2.0f * PI) / f32(pc.phiSteps);

	f32v3 sum = f32v3(0.0f);
	for (u32 ti = 0u; ti < pc.thetaSteps; ++ti) {
		const f32 theta = (f32(ti) + 0.5f) * dTheta;
		const f32 cosThetaO = cos(theta);
		const f32 sinThetaO = sin(theta);
		for (u32 pi = 0u; pi < pc.phiSteps; ++pi) {
			const f32 phi = (f32(pi) + 0.5f) * dPhi;
			const f32v3 wo = (
				f32v3(sinThetaO * cos(phi), sinThetaO * sin(phi), cosThetaO)
			);
			const f32v3 f = openPbrFresnelMetallicEvaluateF(mat, frame, wi, wo);
			sum += f * cosThetaO * sinThetaO * dTheta * dPhi;
		}
	}
	((f32v3 *)(pc.outVa))[i] = sum;
}
