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
	u32 tableHandle;
	u32 thetaSteps;
	u32 phiSteps;
	u32 count;
};

[[vk::push_constant]] PC pc;

// furnace test: numerically integrates openPbrFuzzEvaluateF over the wo
// hemisphere (outgoing directions) for a FIXED wi, under constant unit
// illumination -- the directional-hemispherical reflectance R(wi). unlike
// the glossy-diffuse layer furnace (which fixes wo because its directional
// albedo Rs is defined at NdotV=dot(nor,wo)), the fuzz layer's directional
// albedo eFuzz is the table's own R channel, defined at mu=dot(nor,wi), so
// fixing wi here is what makes eFuzz(wi) factor cleanly out of the integral:
//   integral_wo [weight*fFuzz(wi,wo) + (1-weight*eFuzz(wi))*coatedBase] dwo
//     = weight * R_fuzz_actual(wi) + (1-weight*eFuzz(wi)) * (coatedBase * pi)
// for a constant (wo-independent) coatedBase, since integral_wo cos(theta_o)
// dwo = pi over the hemisphere. with coatedBase=0 and weight=1 this reduces
// to R_fuzz_actual(wi), which the host can compare directly against the
// table's own claimed eFuzz(wi) -- ties the table data to the actual sampled
// BRDF formula it's supposed to be the directional albedo of, not just an
// isolated table crosscheck.

static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per case, 11 floats in: wi.xyz, fuzzWeight, fuzzColor.rgb, fuzzRoughness,
// coatedBase.rgb (canonical local frame, nor = (0,0,1)). output 3 floats:
// the integral, per channel.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 11u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32 fuzzWeight = inBuf[base+3u];
	const f32v3 fuzzColor = (
		f32v3(inBuf[base+4u], inBuf[base+5u], inBuf[base+6u])
	);
	const f32 fuzzRoughness = inBuf[base+7u];
	const f32v3 coatedBase = (
		f32v3(inBuf[base+8u], inBuf[base+9u], inBuf[base+10u])
	);

	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = pc.tableHandle;

	OpenPbrMaterial mat;
	mat.fuzzWeight = fuzzWeight;
	mat.fuzzColor = fuzzColor;
	mat.fuzzRoughness = fuzzRoughness;

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);

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
			const f32v3 f = (
				openPbrFuzzEvaluateF(tables, mat, coatedBase, nor, wi, wo)
			);
			sum += f * cosThetaO * sinThetaO * dTheta * dPhi;
		}
	}
	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*3u+0u] = sum.x;
	outBuf[i*3u+1u] = sum.y;
	outBuf[i*3u+2u] = sum.z;
}
