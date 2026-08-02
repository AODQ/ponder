import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-fuzz";
#include "../util-microfacet-materialx-reference.slang"
#include "../util-fuzz-materialx-reference.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 tableHandle;
	u32 count;
};

[[vk::push_constant]] PC pc;

// cross-check of the real 32x32 fit table (utilZeltnerFuzzLookup, backed by
// ponder::zeltner_tables_create's literal data ported from cull) against
// materialx's independent closed-form rational fits to the same underlying
// zeltner et al. 2022 supplemental data (mx_zeltner_sheen_ltc_aInv/bInv,
// mx_zeltner_sheen_dir_albedo). the two are different fits to the same
// paper's numbers -- a 32x32 tabulation vs. a low-order rational function --
// so exact agreement isn't expected, but they should track closely except
// near roughness -> 0 where the tabulated fit is sparse/zero-padded (see
// the table data comment: the smoothest row is almost entirely zero).

static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per case, 2 floats in: mu, roughness. output 6 floats: tableA, tableB,
// tableR, refAInv, refBInv, refR.

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
	const f32 refAInv = mx_zeltner_sheen_ltc_aInv(mu, roughness);
	const f32 refBInv = mx_zeltner_sheen_ltc_bInv(mu, roughness);
	const f32 refR = mx_zeltner_sheen_dir_albedo(mu, roughness);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*6u+0u] = abr.x;
	outBuf[i*6u+1u] = abr.y;
	outBuf[i*6u+2u] = abr.z;
	outBuf[i*6u+3u] = refAInv;
	outBuf[i*6u+4u] = refBInv;
	outBuf[i*6u+5u] = refR;
}
