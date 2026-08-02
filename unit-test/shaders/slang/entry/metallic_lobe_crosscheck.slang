import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-metallic";
#include "../util-fresnel-materialx-reference.slang"
#include "../util-microfacet-materialx-reference.slang"
import "util-material-openpbr-shared";

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

// full-lobe independent cross-check against a pure-materialx assembly of
// the same conductor model:
//   ref = F_hoffman-schlick * D_mx * G2_mx / (4 muI muO)
//         * (1 + fAvg (1 - E_mx)/E_mx)
// restricted to MIRROR configurations (wo = wi reflected about the
// normal, so the half vector is exactly the normal): ponder evaluates its
// f82 fresnel at dot(nor, wi) while materialx's conductor convention is
// dot(h, wi) -- on mirror configs the two cosines coincide, making the
// comparison exact rather than convention-dependent. isotropic only
// (anisotropy pinned to 0), since mx_ggx_smith_G2 is the scalar-alpha
// form. the fAvg formula (kulla & conty) is shared by both sides; every
// other factor comes from the materialx reference files.
//
// per sample, 9 floats: mu (= dotNorWi = dotNorWo), specularRoughness,
// baseWeight, baseColor.rgb, specularWeight, specularColor (scalar,
// splatted), unused

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 9u;
	const f32 mu = inBuf[base+0u];
	const f32 rough = inBuf[base+1u];

	OpenPbrMaterial mat;
	mat.baseWeight = inBuf[base+2u];
	mat.baseColor = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	mat.specularWeight = inBuf[base+6u];
	mat.specularColor = f32v3(inBuf[base+7u]);
	mat.specularRoughness = rough;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.coatWeight = 0.0f;
	mat.coatRoughness = 0.0f;
	mat.thinFilmWeight = 0.0f;

	ShadingFrame frame;
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	frame.nor = f32v3(0.0f, 0.0f, 1.0f);

	// mirror configuration about the normal
	const f32 s = sqrt(max(0.0f, 1.0f - mu * mu));
	const f32v3 wi = f32v3(s, 0.0f, mu);
	const f32v3 wo = f32v3(-s, 0.0f, mu);

	const f32v3 mine = openPbrFresnelMetallicEvaluateF(mat, frame, wi, wo);

	// materialx-side assembly; alpha floored identically to
	// openPbrRoughnessAlpha's own 1e-5 floor at anisotropy 0
	const f32 alpha = max(rough, 1e-5f);
	const f32v3 f0 = mat.baseWeight * mat.baseColor;
	// clamped to match ponder's deliberate f82 >= 0 deviation (see
	// util-material-openpbr-metallic.glsl); materialx's own
	// hoffman-schlick returns the raw, possibly-negative value
	const f32v3 fresnel = (
		max(
			mxFresnelHoffmanSchlick(
				mu, f0, mat.specularWeight * mat.specularColor, f32v3(1.0f), 5.0f
			),
			f32v3(0.0f)
		)
	);
	const f32 dMx = mx_ggx_NDF(f32v3(0.0f, 0.0f, 1.0f), f32v2(alpha));
	const f32 g2Mx = mx_ggx_smith_G2(mu, mu, alpha);
	// same 1e-4 floor as the production lobe's own ess guard
	const f32 essMx = (
		max(mx_ggx_dir_albedo_analytic(mu, alpha, 1.0f, 1.0f), 1e-4f)
	);
	const f32v3 fAvg = f0 + (f32v3(1.0f) - f0) * (1.0f / 21.0f);
	const f32v3 compMx = f32v3(1.0f) + fAvg * (1.0f - essMx) / essMx;
	const f32v3 ref = (
		fresnel * dMx * (g2Mx / (4.0f * mu * mu)) * compMx
	);

	((f32v3 *)(pc.mineOutVa))[i] = mine;
	((f32v3 *)(pc.refOutVa))[i] = ref;
}
