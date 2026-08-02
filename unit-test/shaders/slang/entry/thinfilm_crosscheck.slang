import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
#include "../util-thinfilm-materialx-reference.slang"
import "util-material-openpbr-shared";
import bsdf;
import material;

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 mineOutVa;
	u64 refOutVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// independent cross-check against materialx's mx_fresnel_airy, restricted
// to a real (non-absorbing) base -- kappaBase = 0 on both sides, matching
// how openPbrThinfilmFresnel is actually called in
// util-material-openpbr-metallic.glsl/-microfacet.glsl (both pass an
// effective real ior base, never a complex one -- see thinfilm_evaluate's
// own conductor-base coverage for the any(kappaBase != 0) branch in isolation).
//
// thickness is restricted to >= 0.03 (see push data) so ponder's
// smoothstep(0, 0.03, thickness) fade-in of eta2 is fully saturated to
// thinFilmIor, matching materialx's unconditional eta2 = max(tf_ior, 1) --
// below that fade-in band the two are not directly comparable (materialx
// has no equivalent ramp).
//
// per sample, 4 floats: cosThetaI, thinFilmThickness (micrometers, >=
// 0.03), thinFilmIor, etaBase (scalar, splatted -- an achromatic base,
// same as the dielectric-specular/microfacet caller)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 4u;
	const f32 cosThetaI = inBuf[base+0u];
	const f32 thickness = inBuf[base+1u];
	const f32 filmIor = inBuf[base+2u];
	const f32 etaBaseScalar = inBuf[base+3u];

	var params: vkof::MaterialBackend_OpenPbr.Params;
	params.thinFilmThickness = thickness;
	params.thinFilmIor = filmIor;

	((f32v3 *)(pc.mineOutVa))[i] = (
		vkof::bsdfOpenPbrThinfilmFresnel(
			params, cosThetaI, f32v3(etaBaseScalar), f32v3(0.0f)
		)
	);
	((f32v3 *)(pc.refOutVa))[i] = (
		mxThinfilmFresnel(
			cosThetaI, thickness * 1000.0f, filmIor,
			f32v3(etaBaseScalar), f32v3(0.0f)
		)
	);
}
