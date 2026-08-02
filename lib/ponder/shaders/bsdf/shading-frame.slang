implementing bsdf;

#include "include/differentiable.slang"

namespace vkof {

public func calculateTbnBasis(nor: f32v3, tangent: f32v4) -> f32m3x3 {
	// gram-schmidt re-orthogonalization
	const f32v3 n = normalize(nor);
	const f32v3 tRaw = tangent.xyz - n * dot(n, tangent.xyz);
	const f32 tLen = length(tRaw);
	if (tLen < 1e-6) {
		// degerate tangent, fall-back to Frisvad basis
		f32v3 binormal, bitangent;
		vkof::calculateXy(n, binormal, bitangent);
		return transpose(f32m3x3(binormal, bitangent, n));
	}
	const f32v3 t = tRaw / tLen;
	const f32v3 bitangent = (cross(n, t) * tangent.w);
	return transpose(f32m3x3(t, bitangent, n));
}

// shading frame for the anisotropic lobes; tanY = cross(nor, tanX).
public struct ShadingFrame : IDifferentiable {
	public f32v3 tanX;
	public f32v3 tanY;
	public f32v3 nor;
};

// frisvad fallback frame; rotationally arbitrary around nor, so only valid
// for isotropic lobes or when no mesh tangent exists
PonderDifferentiable
public func shadingFrameFromNormal(nor: f32v3) -> ShadingFrame {
	f32v3 binormal, bitangent;
	vkof::calculateXy(nor, binormal, bitangent);
	ShadingFrame frame;
	frame.tanX = bitangent;
	frame.tanY = binormal;
	frame.nor = nor;
	return frame;
}

// shading frame from the mesh/uv tangent, re-orthogonalized against the
// (possibly normal-mapped) shading normal
public func shadingFrameFromTangent(
	nor: f32v3,
	tangent: f32v3,
) -> ShadingFrame {
	const f32v3 tRaw = tangent - nor * dot(nor, tangent);
	const f32 tLen2 = dot(tRaw, tRaw);
	if (tLen2 < 1e-10f) {
		// degenerate tangent: fall back to the frisvad frame
		return shadingFrameFromNormal(nor);
	}
	ShadingFrame frame;
	frame.tanX = tRaw * rsqrt(tLen2);
	frame.tanY = cross(nor, frame.tanX);
	frame.nor = nor;
	return frame;
}

// falls back to the geometric normal on a degenerate (zero/NaN-length)
// shading normal, e.g. a missing NORMAL attribute or antiparallel corners
public func shadingNormalOrGeometric(
	shadingNormalRaw: f32v3,
	normalGeometrical: f32v3,
) -> f32v3 {
	const f32 len2 = dot(shadingNormalRaw, shadingNormalRaw);
	if (len2 > 1e-12f) {
		return shadingNormalRaw * rsqrt(len2);
	}
	return normalGeometrical;
}

// raises a shading normal toward the geometric normal just enough that the
// specular reflection of ωo stays above the surface
// keller et al. 2017 (iray)
// cycles' ensure_valid_specular_reflection (bsdf_util.h, bsd-3).
// ωo and geometricNormal on the same side
public func shadingNormalBendToReflect(
	nor: f32v3,
	geometricNormal: f32v3,
	ωo: f32v3,
) -> f32v3 {
	let reflected: f32v3 = 2.0f * dot(nor, ωo) * nor - ωo;
	let oz: f32 = dot(ωo, geometricNormal);
	// a reflection may be as shallow as the incoming ray
	let threshold: f32 = min(0.9f * oz, 0.01f);
	if (dot(geometricNormal, reflected) >= threshold) { return nor; }

	// -- frame: z = geometric normal, nor in the x-z plane
	let xRaw: f32v3 = nor - dot(nor, geometricNormal) * geometricNormal;
	let xLen2: f32 = dot(xRaw, xRaw);
	let x: f32v3 = xLen2 > 1e-12f ? xRaw * rsqrt(xLen2) : nor;
	let ox: f32 = dot(ωo, x);

	// -- solve 4a nz^4 - 2b nz^2 + c = 0 for the root where the reflection
	// sits at the threshold
	let a: f32 = ox * ox + oz * oz;
	let b: f32 = 2.0f * (a + oz * threshold);
	let c: f32 = (threshold + oz) * (threshold + oz);
	let root: f32 = sqrt(max(b * b - 4.0f * a * c, 0.0f));
	let nz2: f32 = 0.25f * (ox < 0.0f ? b + root : b - root) / a;
	let nx: f32 = sqrt(max(1.0f - nz2, 0.0f));
	let nz: f32 = sqrt(max(nz2, 0.0f));
	return nx * x + nz * geometricNormal;
}

// shading frame from a tangent-space shading normal over the mesh basis
public func shadingFrameFromNormalMap(
	meshNormal: f32v3,
	meshTangent: f32v4,
	facing: f32,
	geometricNormal: f32v3,
	mappedNormal: f32v3,
	ωo: f32v3,
) -> ShadingFrame {
	let tbn: f32m3x3 = calculateTbnBasis(meshNormal, meshTangent);
	let mapped: f32v3 = (
		shadingNormalOrGeometric(
			mul(tbn, mappedNormal) * facing, geometricNormal
		)
	);
	let nor: f32v3 = shadingNormalBendToReflect(mapped, geometricNormal, ωo);
	return shadingFrameFromTangent(nor, meshTangent.xyz);
}

// rotates the tangent axes around nor; used for the gltf interop
// anisotropy rotation (khr_materials_anisotropy), which openpbr itself
// does not have. counter-clockwise from the tangent toward the bitangent
/*
	t' = \cos(\phi) t + \sin(\phi) b\\
	b' = \cos(\phi) b - \sin(\phi) t
*/
public func shadingFrameRotate(
	frame: ShadingFrame,
	rotation: f32
) -> ShadingFrame {
	if (rotation == 0.0f) {
		return frame;
	}
	const f32 c = cos(rotation);
	const f32 s = sin(rotation);
	ShadingFrame rotated;
	rotated.tanX = c * frame.tanX + s * frame.tanY;
	rotated.tanY = c * frame.tanY - s * frame.tanX;
	rotated.nor = frame.nor;
	return rotated;
}

} // namespace vkof
