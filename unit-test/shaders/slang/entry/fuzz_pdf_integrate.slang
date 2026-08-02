import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-fuzz";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per case, 8 floats in: nor.xyz, wi.xyz, fuzzA, fuzzB. output 1 float: the
// pdf integrated over the full sphere of wo directions by midpoint
// quadrature, clamped to 0 below the horizon the same way production code
// clamps openPbrGlossyDiffusePdf (the sampler only ever emits above the
// nor-horizon; see the "always upper hemisphere" derivation in the fuzz
// sampler doc comment -- openPbrFuzzPdf itself has no clamp, matching the
// existing diffuse pdf's convention of leaving that to the caller). a valid
// pdf integrates to 1.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 8u;
	const f32v3 nor = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 wi = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32 fuzzA = inBuf[base+6u];
	const f32 fuzzB = inBuf[base+7u];

	const u32 kThetaSteps = 1024u;
	const u32 kPhiSteps = 512u;
	const f32 dTheta = PI / f32(kThetaSteps);
	const f32 dPhi = TAU / f32(kPhiSteps);

	f32v3 tanX, tanY;
	utilCalculateXy(nor, tanX, tanY);

	f32 integral = 0.0f;
	for (u32 ti = 0u; ti < kThetaSteps; ++ti) {
		const f32 theta = (f32(ti) + 0.5f) * dTheta;
		const f32 sinTheta = sin(theta);
		const f32 cosTheta = cos(theta);
		for (u32 pi_ = 0u; pi_ < kPhiSteps; ++pi_) {
			const f32 phi = (f32(pi_) + 0.5f) * dPhi;
			const f32v3 wo = (
				tanX * (sinTheta * cos(phi))
				+ tanY * (sinTheta * sin(phi))
				+ nor * cosTheta
			);
			const f32 pdf = openPbrFuzzPdf(nor, wi, wo, fuzzA, fuzzB);
			integral += max(pdf, 0.0f) * sinTheta;
		}
	}
	integral *= dTheta * dPhi;

	((f32 *)(pc.outVa))[i] = integral;
}
