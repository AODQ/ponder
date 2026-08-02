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
	u32 count;
};

[[vk::push_constant]] PC pc;

// evaluate shader for the ported layered fuzz eval (openPbrFuzzEvaluateF),
// using the real zeltner ltc table.

static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per sample, 17 floats in: nor.xyz, wi.xyz, wo.xyz, fuzzWeight,
// fuzzColor.rgb, fuzzRoughness, coatedBase.rgb. output 3 floats:
// openPbrFuzzEvaluateF(tables, mat, coatedBase, nor, wi, wo).

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 17u;
	const f32v3 nor = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 wi = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32v3 wo = (
		f32v3(inBuf[base+6u], inBuf[base+7u], inBuf[base+8u])
	);
	const f32 fuzzWeight = inBuf[base+9u];
	const f32v3 fuzzColor = (
		f32v3(inBuf[base+10u], inBuf[base+11u], inBuf[base+12u])
	);
	const f32 fuzzRoughness = inBuf[base+13u];
	const f32v3 coatedBase = (
		f32v3(inBuf[base+14u], inBuf[base+15u], inBuf[base+16u])
	);

	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = pc.tableHandle;

	OpenPbrMaterial mat;
	mat.fuzzWeight = fuzzWeight;
	mat.fuzzColor = fuzzColor;
	mat.fuzzRoughness = fuzzRoughness;

	const f32v3 f = (
		openPbrFuzzEvaluateF(tables, mat, coatedBase, nor, wi, wo)
	);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*3u+0u] = f.x;
	outBuf[i*3u+1u] = f.y;
	outBuf[i*3u+2u] = f.z;
}
