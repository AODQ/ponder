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

// evaluate shader for the ported zeltner ltc parameter table lookup
// (utilZeltnerFuzzLookup): bilinear fetch from the real 32x32 texture
// (ponder::zeltner_tables_create) at (mu, roughness).

static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per case, 2 floats in: mu, roughness. output 3 floats: (a, b, R).

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	const f32 mu = inBuf[i*2u+0u];
	const f32 roughness = inBuf[i*2u+1u];

	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = pc.tableHandle;

	const f32v3 abr = utilZeltnerFuzzLookup(tables, mu, roughness);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*3u+0u] = abr.x;
	outBuf[i*3u+1u] = abr.y;
	outBuf[i*3u+2u] = abr.z;
}
