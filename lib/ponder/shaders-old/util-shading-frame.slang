module "util-shading-frame";
__exported import "util-types";
__exported import "util-random";

public func utilCalculateXy(
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

// deviates from the glsl only in column order: slang's f32m3x3(a, b, c)
// fills rows, glsl's mat3(a, b, c) fills columns
public func utilCalculateTbnBasis(nor: f32v3, tangent: f32v4) -> f32m3x3 {
	// gram-schmidt re-orthogonalization
	const f32v3 n = normalize(nor);
	const f32v3 tRaw = tangent.xyz - n * dot(n, tangent.xyz);
	const f32 tLen = length(tRaw);
	if (tLen < 1e-6) {
		// degerate tangent, fall-back to Frisvad basis
		f32v3 binormal, bitangent;
		utilCalculateXy(n, binormal, bitangent);
		return transpose(f32m3x3(binormal, bitangent, n));
	}
	const f32v3 t = tRaw / tLen;
	const f32v3 bitangent = (cross(n, t) * tangent.w);
	return transpose(f32m3x3(t, bitangent, n));
}

// shading frame for the anisotropic lobes; tanY = cross(nor, tanX).
// bitangent handedness does not matter distribution-wise: the aniso ggx D
// and smith lambda are even in both tangent axes, and the cap sampler's
// azimuth is distribution-uniform
public struct ShadingFrame {
	public f32v3 tanX;
	public f32v3 tanY;
	public f32v3 nor;
};

// frisvad fallback frame; rotationally arbitrary around nor, so only valid
// for isotropic lobes or when no mesh tangent exists
public func shadingFrameFromNormal(nor: f32v3) -> ShadingFrame {
	f32v3 binormal, bitangent;
	utilCalculateXy(nor, binormal, bitangent);
	ShadingFrame frame;
	frame.tanX = bitangent;
	frame.tanY = binormal;
	frame.nor = nor;
	return frame;
}

// shading frame from the mesh/uv tangent, re-orthogonalized against the
// (possibly normal-mapped) shading normal
public func shadingFrameFromTangent(nor: f32v3, tangent: f32v3) -> ShadingFrame {
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
public func utilShadingNormalOrGeometric(
	shadingNormalRaw: f32v3,
	normalGeometrical: f32v3,
) -> f32v3 {
	const f32 len2 = dot(shadingNormalRaw, shadingNormalRaw);
	if (len2 > 1e-12f) {
		return shadingNormalRaw * rsqrt(len2);
	}
	return normalGeometrical;
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

public func utilReorientHemisphere(wo: f32v3, nor: f32v3) -> f32v3 {
	f32v3 binormal, bitangent;
	const f32v3 n = normalize(nor);
	utilCalculateXy(n, binormal, bitangent);
	return bitangent*wo.x + binormal*wo.y + wo.z*n;
}

public func utilToCartesian(cosTheta: f32, phi: f32) -> f32v3 {
	const f32 sinTheta = sqrt(max(0.0f, 1.0f - cosTheta*cosTheta));
	return f32v3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
}

public func utilCosineHemisphereSampleWo(nor: f32v3, xi: f32v2) -> f32v3 {
	const f32v3 wo = (
		utilReorientHemisphere(
			normalize(utilToCartesian(sqrt(xi.x), skTau*xi.y)),
			nor
		)
	);
	return wo;
}

public func utilDirectionAboutAxis(
	axis: f32v3, cosTheta: f32, phi: f32,
) -> f32v3 {
	const ShadingFrame frame = shadingFrameFromNormal(axis);
	const f32 sinTheta = sqrt(max(0.0f, 1.0f - cosTheta*cosTheta));
	return (
		frame.tanX * (sinTheta * cos(phi))
		+ frame.tanY * (sinTheta * sin(phi))
		+ frame.nor * cosTheta
	);
}
