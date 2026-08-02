import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
	u32 thetaSteps;
	u32 phiSteps;
};

[[vk::push_constant]] PC pc;

// hemispherical integrals of the thin-walled subsurface sheet, one thread
// per config: the reflection albedo E_R, the transmission albedo E_T (which
// must sum to S per the spec's energy statement), and the pdf's own
// sphere normalization.

static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per config, 8 floats in: wi.xyz, baseDiffuseRoughness,
// subsurfaceScatterAnisotropy, subsurfaceColor.rgb. 8 floats out:
// E_R.rgb, E_T.rgb, pdf sphere integral, unused. nor is fixed to (0,0,1)

[shader("compute")]
[numthreads(16, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) {
		return;
	}
	f32 * inBuf = (f32 *)(pc.inVa);
	const u32 base = i * 8u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);

	OpenPbrMaterial mat;
	mat.baseDiffuseRoughness = inBuf[base+3u];
	mat.subsurfaceScatterAnisotropy = inBuf[base+4u];
	mat.subsurfaceColor = (
		f32v3(inBuf[base+5u], inBuf[base+6u], inBuf[base+7u])
	);

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	// midpoint quadrature over the whole sphere; the upper/lower split is
	// decided per sample by the sign of cos\theta
	const f32 dTheta = PI / f32(pc.thetaSteps);
	const f32 dPhi = TAU / f32(pc.phiSteps);
	f32v3 albedoReflect = f32v3(0.0f);
	f32v3 albedoTransmit = f32v3(0.0f);
	f32 pdfIntegral = 0.0f;
	for (u32 t = 0u; t < pc.thetaSteps; ++t) {
		const f32 theta = (f32(t) + 0.5f) * dTheta;
		const f32 cosTheta = cos(theta);
		const f32 sinTheta = sin(theta);
		for (u32 p = 0u; p < pc.phiSteps; ++p) {
			const f32 phi = (f32(p) + 0.5f) * dPhi;
			const f32v3 wo = (
				f32v3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta)
			);
			const f32 solidAngle = sinTheta * dTheta * dPhi;
			const f32v3 f = (
				openPbrThinWalledSubsurfaceEvaluateF(mat, nor, wi, wo)
			);
			const f32v3 contribution = f * abs(cosTheta) * solidAngle;
			if (cosTheta < 0.0f) {
				albedoTransmit += contribution;
			} else {
				albedoReflect += contribution;
			}
			pdfIntegral += (
				openPbrThinWalledSubsurfacePdf(
					cosTheta, mat.subsurfaceScatterAnisotropy
				)
				* solidAngle
			);
		}
	}

	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 outBase = i * 8u;
	outBuf[outBase+0u] = albedoReflect.x;
	outBuf[outBase+1u] = albedoReflect.y;
	outBuf[outBase+2u] = albedoReflect.z;
	outBuf[outBase+3u] = albedoTransmit.x;
	outBuf[outBase+4u] = albedoTransmit.y;
	outBuf[outBase+5u] = albedoTransmit.z;
	outBuf[outBase+6u] = pdfIntegral;
	outBuf[outBase+7u] = 0.0f;
}
