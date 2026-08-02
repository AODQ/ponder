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
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per sample, 3 floats in: specularIor, transmissionDispersionScale,
// transmissionDispersionAbbeNumber. output 3 floats: openPbrDispersionIorRgb

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 3u;

	OpenPbrMaterial mat;
	mat.specularIor = inBuf[base+0u];
	// n_d is now the coat-affected, specular_weight-modulated ior; these are
	// its identities, so the golden values stay the plain-specularIor cauchy fit
	mat.specularWeight = 1.0f;
	mat.coatWeight = 0.0f;
	mat.coatIor = 1.0f;
	mat.transmissionDispersionScale = inBuf[base+1u];
	mat.transmissionDispersionAbbeNumber = inBuf[base+2u];

	const f32v3 iorRgb = openPbrDispersionIorRgb(mat);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*3u+0u] = iorRgb.x;
	outBuf[i*3u+1u] = iorRgb.y;
	outBuf[i*3u+2u] = iorRgb.z;
}
