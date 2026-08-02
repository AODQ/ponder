implementing bsdf;

import material;
import util;
import types;

namespace vkof {

// dielectric fresnel
// η = η for reflection, 1/η for transmission
// cosθ >= 0
public func bsdfFresnelDielectric(cosθ: f32, η: f32) -> f32 {
	let cosθ2: f32 = cosθ * cosθ;
	let sinθ2: f32 = 1.0f - cosθ2;
	let η2: f32 = η * η;
	let sinθt2: f32 = sinθ2 / η2;
	// total internal reflection
	if (sinθt2 >= 1.0f) { return 1.0f; }
	let cosθt: f32 = sqrt(1.0f - sinθt2);

	// -- calculate the parallel and perpendicular reflection coefficients
	// at cosθ=0, η=1, both denominators approach 0/0
	let rParallelDenom = max(1e-7f, η * cosθ + cosθt);
	let rParallel: f32 = (η * cosθ - cosθt) / rParallelDenom;
	let rPerpendicularDenom = max(1e-7f, η * cosθt + cosθ);
	let rPerpendicular: f32 = (cosθ - η * cosθt) / rPerpendicularDenom;
	return 0.5f * (rParallel*rParallel + rPerpendicular*rPerpendicular);
}

// f82-tint conductor fresnek
// kutz, hasan & edmondson 2021
// cosθ must be >= 0
public func bsdfFresnelMetallic(
	f0: vkof::Spectrum,
	specularWeight: f32,
	specularColor: vkof::Spectrum,
	cosθ: f32,
) -> vkof::Spectrum
{
	// F_s(1/7); (1 - 1/7)^5 = 0.462664
	let skOne: vkof::Spectrum = vkof::Spectrum.one();
	let skZero: vkof::Spectrum = vkof::Spectrum.zero();
	let fresnelConstS = f0 + (skOne - f0) * 0.462664f;
	// (1/7)(1 - 1/7)^6 = 0.056653
	let fresnelDenom = 0.056653f;
	let fs = f0 + (skOne - f0) * pow(1.0f - cosθ, 5.0f);

	return (
		max(
			fs
			- (
				(cosθ * pow(1.0f - cosθ, 6.0f) / fresnelDenom)
				* fresnelConstS
				* (skOne - specularWeight * specularColor)
			),
			skZero
		)
	);
}


} // namespace vkof
