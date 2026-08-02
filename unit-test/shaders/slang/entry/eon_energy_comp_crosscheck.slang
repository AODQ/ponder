import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
#include "../util-eon-materialx-reference.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the diffuse-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per sample: f32v2(mu, sigma). output f32v2(mine, materialx) where mine is
// the portsmouth polynomial fit and materialx is the exact closed-form
// fujii directional albedo E_FON

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v2 s = ((f32v2 *)(pc.inVa))[i];
	const f32 mine = utilOrenNayerEnergyCompensate(s.x, s.y);
	const f32 ref = mxOrenNayarFujiiDiffuseDirAlbedo(s.x, s.y);
	((f32v2 *)(pc.outVa))[i] = f32v2(mine, ref);
}
