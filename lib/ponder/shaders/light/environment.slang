implementing light;

import types;
import util;

// TODO this file needs review

namespace vkof {

// mirrors ponder::WavefrontEnvironmentMode
public enum EnvironmentMode : u32 {
	Furnace = 0u,
	Checkerboard = 1u,
	Black = 2u,
	HdrMap = 3u,
};

// distance at which a path counts as having escaped to the environment
public static const f32 skEnvironmentDistance = 5000.0f;

// a direction sampled from the environment
public struct EnvironmentSample {
	public f32v3 ωo;
	// solid-angle pdf of ωo; 0 when the environment can't be sampled
	public f32 pdf;
};

// radiance arriving from the environment along dir
public func environmentRadiance(
	dir: f32v3,
	λ: vkof::Wavelengths,
) -> vkof::Spectrum
{
	let intensity: f32 = vkof::global.environmentIntensity;
	switch (EnvironmentMode(vkof::global.environmentMode)) {
		case EnvironmentMode.HdrMap: {
			// bilinear radiance against an exact-texel pdf, as pbrt does
			let rgb: f32v3 = (
				vkofTextures[
					NonUniformResourceIndex(
						vkof::global.environmentRadianceHandle
					)
				]
				.SampleLevel(implEnvironmentDirToUv(dir), 0.0f)
				.rgb
			);
			return vkof::ColorEmission(rgb, intensity).toSpectrum(λ);
		}
		case EnvironmentMode.Checkerboard: {
			let checkerSize: f32 = 0.1f;
			let cell: f32 = (
				floor(dir.x / checkerSize) + floor(dir.z / checkerSize)
			);
			let checker: f32 = cell - 2.0f * floor(cell * 0.5f);
			let value: f32 = lerp(0.2f, 1.0f, checker);
			return (
				vkof::ColorEmission(f32v3(value), intensity).toSpectrum(λ)
			);
		}
		case EnvironmentMode.Black: {
			return vkof::Spectrum::zero();
		}
		case EnvironmentMode.Furnace:
		default: {
			return vkof::ColorEmission(f32v3(1.0f), intensity).toSpectrum(λ);
		}
	}
}

// solid-angle pdf environmentSample has of producing dir
public func environmentPdf(dir: f32v3) -> f32 {
	if (
		EnvironmentMode(vkof::global.environmentMode)
		!= EnvironmentMode.HdrMap
	) {
		return 0.0f;
	}
	let uv: f32v2 = implEnvironmentDirToUv(dir);
	let sinθ: f32 = sin(uv.y * π);
	if (sinθ <= 1e-6f) { return 0.0f; }
	// -- uv to the texel holding its probability
	let handle: u32 = vkof::global.environmentPdfHandle;
	let size: u32v2 = implEnvironmentSize(handle);
	let texel: u32v2 = min(u32v2(uv * f32v2(size)), size - u32v2(1u));
	// -- texel probability to uv density, then the equirect jacobian
	let pdfUv: f32 = (
		implEnvironmentTexel(handle, texel) * f32(size.x) * f32(size.y)
	);
	return pdfUv / (2.0f * π * π * sinθ);
}

// importance-samples a direction by vose's alias method
// ξ.x picks a texel uniformly, ξ.y's comparison against the alias table
// redirects it; what is left of both becomes the sub-texel jitter
public func environmentSample(ξ: f32v2) -> EnvironmentSample {
	var sample: EnvironmentSample;
	sample.ωo = f32v3(0.0f, 1.0f, 0.0f);
	sample.pdf = 0.0f;
	if (
		EnvironmentMode(vkof::global.environmentMode)
		!= EnvironmentMode.HdrMap
	) {
		return sample;
	}
	let probHandle: u32 = vkof::global.environmentAliasProbHandle;
	let size: u32v2 = implEnvironmentSize(probHandle);
	let count: u32 = size.x * size.y;
	if (count == 0u) { return sample; }

	// -- pick a texel uniformly
	let scaled: f32 = ξ.x * f32(count);
	let index: u32 = min(u32(scaled), count - 1u);
	let jitterX: f32 = scaled - f32(index);
	let texel: u32v2 = u32v2(index % size.x, index / size.x);

	// -- keep it or take the texel its bucket redirects to
	let threshold: f32 = implEnvironmentTexel(probHandle, texel);
	var finalIndex: u32 = index;
	var jitterY: f32 = ξ.y / max(threshold, 1e-8f);
	if (ξ.y >= threshold) {
		finalIndex = (
			u32(
				implEnvironmentTexel(
					vkof::global.environmentAliasIndexHandle, texel
				)
				+ 0.5f
			)
		);
		jitterY = (ξ.y - threshold) / max(1.0f - threshold, 1e-8f);
	}

	// -- continuous uv to direction
	let uv: f32v2 = (
		f32v2(
			(f32(finalIndex % size.x) + jitterX) / f32(size.x),
			(f32(finalIndex / size.x) + jitterY) / f32(size.y)
		)
	);
	sample.ωo = implEnvironmentUvToDir(uv);
	sample.pdf = environmentPdf(sample.ωo);
	return sample;
}

} // vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

// world-space direction to equirect uv, y up
func implEnvironmentDirToUv(dir: f32v3) -> f32v2 {
	let rotation: f32 = vkof::global.environmentRotation;
	let θ: f32 = acos(clamp(dir.y, -1.0f, 1.0f));
	let φ: f32 = atan2(dir.z, dir.x);
	return (
		f32v2(
			frac(0.5f + φ / (2.0f * π) + rotation / (2.0f * π)),
			θ / π
		)
	);
}

// inverse of implEnvironmentDirToUv
func implEnvironmentUvToDir(uv: f32v2) -> f32v3 {
	let rotation: f32 = vkof::global.environmentRotation;
	let θ: f32 = uv.y * π;
	let φ: f32 = (uv.x - 0.5f - rotation / (2.0f * π)) * 2.0f * π;
	return f32v3(sin(θ) * cos(φ), cos(θ), sin(θ) * sin(φ));
}

func implEnvironmentSize(handle: u32) -> u32v2 {
	var width: u32 = 0u;
	var height: u32 = 0u;
	vkofTextures[NonUniformResourceIndex(handle)].GetDimensions(width, height);
	return u32v2(width, height);
}

func implEnvironmentTexel(handle: u32, texel: u32v2) -> f32 {
	return (
		vkofTextures[NonUniformResourceIndex(handle)]
		.Load(i32v3(i32v2(texel), 0))
		.r
	);
}
