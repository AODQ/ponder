implementing bsdf;

#include "include/differentiable.slang"

import types;

// -----------------------------------------------------------------------------
// -- references
// -----------------------------------------------------------------------------
// [F12]   frisvad 2012, "building an orthonormal basis from a 3d unit vector
//         without normalization", jgt vol 16 no 3
// [DBC17] duff, burgess, christensen, hery, kensler, liani & villemin 2017,
//         "building an orthonormal basis, revisited", jcgt vol 6 no 1
//         https://jcgt.org/published/0006/01/01/
// -----------------------------------------------------------------------------

namespace vkof {

// -----------------------------------------------------------------------------
// -- orthonormal basis
// -----------------------------------------------------------------------------

// [F12] with the sign fix of [DBC17]
PonderDifferentiable
public func calculateXy(
	nor: f32v3,
	out binormal: f32v3,
	out bitangent: f32v3,
) {
	// frisvad duff 2017 orthornormal basis revisted
	const f32 sgn = nor.z >= 0.0f ? 1.0f : -1.0f;
	const f32 a = -1.0f / (sgn + nor.z);
	const f32 b = nor.x * nor.y * a;
	binormal = f32v3(1.0f + sgn*nor.x*nor.x*a, sgn*b, -sgn*nor.x);
	bitangent = f32v3(b, sgn + nor.y*nor.y*a, -nor.y);
}

// -----------------------------------------------------------------------------
// -- directions
// -----------------------------------------------------------------------------

// rotates a +z-up local direction into the frame about nor
PonderDifferentiable
public func reorientHemisphere(wo: f32v3, nor: f32v3) -> f32v3 {
	f32v3 binormal, bitangent;
	const f32v3 n = normalize(nor);
	vkof::calculateXy(n, binormal, bitangent);
	return bitangent*wo.x + binormal*wo.y + wo.z*n;
}

// spherical to cartesian, +z is the pole
PonderDifferentiable
public func toCartesian(cosTheta: f32, phi: f32) -> f32v3 {
	const f32 sinTheta = sqrt(max(0.0f, 1.0f - cosTheta*cosTheta));
	return f32v3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
}

// direction at polar angle θ and azimuth φ about axis
PonderDifferentiable
public func directionAboutAxis(
	axis: f32v3,
	cosθ: f32,
	φ: f32,
) -> f32v3 {
	let frame: ShadingFrame = shadingFrameFromNormal(axis);
	const f32 sinθ = sqrt(max(0.0f, 1.0f - cosθ*cosθ));
	return (
		frame.tanX * (cos(φ)*sinθ)
		+ frame.tanY * (sin(φ)*sinθ)
		+ frame.nor * cosθ
	);
}

public func balanceHeuristic(pdfA: f32, pdfB: f32) -> f32 {
	let sum: f32 = pdfA + pdfB;
	if (sum <= 0.0f) { return 0.0f; }
	return pdfA / sum;
}

} // namespace vkof
