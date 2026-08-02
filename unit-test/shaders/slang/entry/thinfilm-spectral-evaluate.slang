import types;
import bsdf;
import material;
import util;

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// spectral vkof::bsdfOpenPbrThinfilmFresnel at four given wavelengths.
// per sample, 8 floats in:
//   cosThetaI, thinFilmThickness (micrometers), thinFilmIor, etaBase,
//   lambda0..3 (nanometers)
// 4 floats out: reflectance per wavelength

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	let i: u32 = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	let inBuf: f32 * = (f32 *)(pc.inVa);
	let base: u32 = i * 8u;

	var params: vkof::MaterialBackend_OpenPbr.Params;
	params.thinFilmThickness = inBuf[base + 1u];
	params.thinFilmIor = inBuf[base + 2u];

	let λ: vkof::Wavelengths = (
		vkof::Wavelengths(
			{
				inBuf[base + 4u],
				inBuf[base + 5u],
				inBuf[base + 6u],
				inBuf[base + 7u],
			},
			{ 1.0f, 1.0f, 1.0f, 1.0f }
		)
	);
	let reflectance: vkof::Spectrum = (
		vkof::bsdfOpenPbrThinfilmFresnel(
			params,
			inBuf[base + 0u],
			vkof::Spectrum.one() * inBuf[base + 3u],
			vkof::Spectrum.zero(),
			λ
		)
	);
	((f32v4 *)(pc.outVa))[i] = (
		f32v4(
			reflectance.channel(0u),
			reflectance.channel(1u),
			reflectance.channel(2u),
			reflectance.channel(3u)
		)
	);
}
