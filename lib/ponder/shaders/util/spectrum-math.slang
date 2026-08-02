implementing util;

import types;

#include "include/inline-expression.slang"

// -----------------------------------------------------------------------------
// -- references
// -----------------------------------------------------------------------------
// [JH19]  jakob & hanika 2019, "a low-dimensional function space for
//         efficient spectral upsampling", eurographics
//         https://rgl.epfl.ch/publications/Jakob2019Spectral
// [WSS13] wyman, sloan & shirley 2013, "simple analytic approximations to
//         the cie xyz color matching functions", jcgt vol 2 no 2
//         https://jcgt.org/published/0002/02/01/
// -----------------------------------------------------------------------------

namespace vkof {

// cie y normalization, ∫ ȳ(λ) dλ in nm; a flat unit spectrum has Y = 1
static const f32 skCieYIntegral = 106.923f;

public T signNonZero<T : __BuiltinFloatingPointType>(T x) {
	return select(x >= T(0.0), T(1.0), T(-1.0));
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

// jakob-hanika upsample evaluated at path's wavelengths
// jakob-hanika 2019
func implRgbToSpectrumFetch(
	rgb: f32v3,
) 
	-> f32v3
{
	// the idea is to use smooth sigmoid-based functions to approximate the
	// reflectance spectrum of a given RGB; the result is a smooth and
	// energy-conserving gradient:
	// s(\lambda) = S(c_0 \lambda^2 + c_1 \lambda + c_2)
	// S(x) = \frac{1}{2} + x / ( 2 * \sqrt{1 + x^2} )
	// (C_0, C_1, C_2) are precomputed per gamut offline
	let resolution: u32 = vkof::global.rgbToSpectrumResolution;
	let rgbClamp: f32v3 = clamp(rgb, 0.0f, 1.0f);

	// if each channel is equal, then the spectrum has a cheap closed-form
	// solution; when used in the S(\lambda) function it will generate a flat
	// line across the spectrum rather than a sigmoid.
	// Note this is required to prevent division by zero for blacks.
	if (rgbClamp.r == rgbClamp.y && rgbClamp.y == rgbClamp.z) {
		let v: f32 = rgbClamp.x;
		// 8192 is a stand-in for a finite infinity that will still valuate to 0/1
		if (v <= 0.0f) { return f32v3(0.0f, 0.0f, -8192.0f); }
		if (v >= 1.0f) { return f32v3(0.0f, 0.0f, 8192.0f); }
		let c: f32 = (v - 0.5f) / sqrt(v * (1.0f - v));
		return f32v3(0.0f, 0.0f, c);
	}
	// note: potential contrast shortcut where r=gb

	// -- fetch components by magnitude, sorted
	let componentIndex: u32 = InlineExpr {
		let i: u32 = rgbClamp.y >= rgbClamp.x ? 1u : 0u;
		return rgbClamp.z >= rgbClamp[i] ? 2u : i;
	};

	// -- fetch the precomputed coefficient slabs
	let indexZ: f32 = rgbClamp[componentIndex];
	let indexYxScale: f32 = f32(resolution - 1u) / indexZ;
	let indexY: f32 = rgbClamp[(componentIndex + 2u) % 3] * indexYxScale;
	let indexX: f32 = rgbClamp[(componentIndex + 1u) % 3] * indexYxScale;
	let indexXi: u32 = min(u32(indexX), resolution - 2u);
	let indexYi: u32 = min(u32(indexY), resolution - 2u);

	// -- binary search z's interval
	// it is a non-uniform monotonic grid
	let indexZi: u32 = InlineExpr {
		var left: u32 = 0u;
		var size: u32 = resolution - 2u;
		while (size > 0u) {
			let half: u32 = size >> 1u;
			let middle: u32 = left + half + 1u;
			if (vkof::global.rgbToSpectrum[middle] <= indexZ) {
				left = middle;
				size -= half + 1u;
			} else {
				size = half;
			}
		}
		return min(left, resolution - 2u);
	};

	// -- fetch the precomputed coefficients
	let coeffs: f32 * = vkof::global.rgbToSpectrum + resolution;
	let offset: u32 = (
		(
			(
				(
					componentIndex * resolution + indexZi
				) * resolution + indexYi
			) * resolution + indexXi
		) * 3u
	);
	let dx: u32 = 3u;
	let dy: u32 = 3u * resolution;
	let dz: u32 = 3u * resolution * resolution;
	let x1: f32 = indexX - f32(indexXi);
	let y1: f32 = indexY - f32(indexYi);
	let z1: f32 = (
		(indexZ - vkof::global.rgbToSpectrum[indexZi])
		/ (
			vkof::global.rgbToSpectrum[indexZi + 1u]
			- vkof::global.rgbToSpectrum[indexZi]
		)
	);
	f32v3 c;
	for (u32 j = 0u; j < 3u; ++j) {
		let o: u32 = offset + j;
		c[j] = (
			lerp(
				lerp(
					lerp(coeffs[o], coeffs[o + dx], x1),
					lerp(coeffs[o + dy], coeffs[o + dy + dx], x1),
					y1
				),
				lerp(
					lerp(coeffs[o + dz], coeffs[o + dz + dx], x1),
					lerp(coeffs[o + dz + dy], coeffs[o + dz + dy + dx], x1),
					y1
				),
				z1
			)
		);
	}
	return c;
}

// -----------------------------------------------------------------------------

func implGaussian(
	lambda: f32, alpha: f32, mu: f32, sigmaLow: f32, sigmaHigh: f32,
)
	-> f32
{
	let t: f32 = (lambda - mu) / (lambda < mu ? sigmaLow : sigmaHigh);
	return alpha * exp(-0.5f * t * t);
}

// -----------------------------------------------------------------------------

// wyman-sloan-shirley fit
// analytic approximatino of the CIE 1931 XYZ
func implCieXyz(lambda: f32) -> f32v3 {
	let x: f32 = (
		implGaussian(lambda, 1.056f, 599.8f, 37.9f, 31.0f)
		+ implGaussian(lambda, 0.362f, 442.0f, 16.0f, 26.7f)
		- implGaussian(lambda, 0.065f, 501.10f, 20.4f, 26.2f)
	);
	let y: f32 = (
		implGaussian(lambda, 0.821f, 568.80f, 46.9f, 40.5f)
		+ implGaussian(lambda, 0.286f, 530.9f, 16.3f, 31.1f)
	);
	let z: f32 = (
		implGaussian(lambda, 1.217f, 437.0f, 11.8f, 36.0f)
		+ implGaussian(lambda, 0.681f, 459.0f, 26.0f, 13.8f)
	);
	return f32v3(x, y, z);
}

func implSpectrumSigmoidUpsample(
	rgb: f32v3,
	λ: vkof::Wavelengths
) -> vkof::SpectrumSampled {
	let c: f32v3 = implRgbToSpectrumFetch(rgb);
	// s(\lambda) = S(c_0 \lambda^2 + c_1 \lambda + c_2)
	// done per-wavelength
	let x: f32[4] = {
		(c.x * λ.lambdas[0] + c.y) * λ.lambdas[0] + c.z,
		(c.x * λ.lambdas[1] + c.y) * λ.lambdas[1] + c.z,
		(c.x * λ.lambdas[2] + c.y) * λ.lambdas[2] + c.z,
		(c.x * λ.lambdas[3] + c.y) * λ.lambdas[3] + c.z,
	};
	return vkof::SpectrumSampled({
		0.5f + 0.5f * x[0] / sqrt(1.0f + x[0] * x[0]),
		0.5f + 0.5f * x[1] / sqrt(1.0f + x[1] * x[1]),
		0.5f + 0.5f * x[2] / sqrt(1.0f + x[2] * x[2]),
		0.5f + 0.5f * x[3] / sqrt(1.0f + x[3] * x[3]),
	});
}
