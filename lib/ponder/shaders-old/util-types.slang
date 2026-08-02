module "util-types";

// slang counterpart of the f32/u32 aliases the glsl shaders #define

public typedef float32_t f32;
public typedef float32_t2 f32v2;
public typedef float32_t3 f32v3;
public typedef float32_t4 f32v4;
public typedef float32_t2x2 f32m2x2;
public typedef float32_t3x3 f32m3x3;
public typedef float32_t4x4 f32m4x4;
public typedef float32_t4x4 f32m44;
// a glsl mat4x3 is 4 columns by 3 rows; slang names a matrix rows first, so
// the same storage is a 3x4
public typedef float32_t3x4 f32m4x3;
public typedef float32_t4x3 f32m3x4;

public typedef int32_t i32;
public typedef int32_t2 i32v2;
public typedef int32_t3 i32v3;
public typedef int32_t4 i32v4;

public typedef uint32_t u32;
public typedef uint32_t2 u32v2;
public typedef uint32_t3 u32v3;
public typedef uint32_t4 u32v4;
public typedef uint64_t u64;

public static const f32 IPI = 0.31830988618379067154;
public static const f32 PI = 3.14159265358979323846;
public static const f32 TAU = 6.28318530717958647692;

// oren-nayar fon coefficients; used across the diffuse lobes
public static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
public static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// vkof binds these at set 0 for every pipeline; declared once here so a
// module that samples does not redeclare and make the name ambiguous
[[vk::binding(0, 0)]] public Sampler2D vkofTextures[];
[[vk::binding(1, 0)]] public RWTexture2D<u32> vkofImagesU[];
[[vk::binding(2, 0)]] public RWTexture2D<f32v4> vkofImages[];
[[vk::binding(3, 0)]] public RaytracingAccelerationStructure vkofTlas;
[[vk::binding(4, 0)]] public Sampler3D vkofTextures3D[];
[[vk::binding(5, 0)]] public RWTexture3D<f32v4> vkofImages3D[];

// glsl-compatible texture shims, so a ported shader keeps the call it had.
// slang spells these as methods, and textureSize has no expression form at
// all -- GetDimensions returns through out parameters
public func texelFetch(tex: Sampler2D, coord: i32v2, lod: i32) -> f32v4 {
	return tex.Load(i32v3(coord, lod));
}

public func textureSize(tex: Sampler2D, lod: i32) -> i32v2 {
	u32 width;
	u32 height;
	u32 levels;
	tex.GetDimensions(u32(lod), width, height, levels);
	return i32v2(i32(width), i32(height));
}

// compute shaders have no derivatives, so glsl's implicit-lod texture() is
// lod 0 here
public func texture(tex: Sampler2D, uv: f32v2) -> f32v4 {
	return tex.SampleLevel(uv, 0.0f);
}

public func textureQueryLevels(tex: Sampler2D) -> i32 {
	u32 width;
	u32 height;
	u32 levels;
	tex.GetDimensions(0u, width, height, levels);
	return i32(levels);
}

public func textureGrad(
	tex: Sampler2D,
	uv: f32v2,
	uvDx: f32v2,
	uvDy: f32v2,
) -> f32v4 {
	return tex.SampleGrad(uv, uvDx, uvDy);
}

public func textureLod(tex: Sampler2D, uv: f32v2, lod: f32) -> f32v4 {
	return tex.SampleLevel(uv, lod);
}

// glsl's image load/store are subscripts in slang
public func imageStore(tex: RWTexture3D<f32v4>, coord: i32v3, value: f32v4) {
	tex[u32v3(coord)] = value;
}

public func imageStore(tex: RWTexture2D<f32v4>, coord: i32v2, value: f32v4) {
	tex[u32v2(coord)] = value;
}

public func imageStore(tex: RWTexture2D<u32>, coord: i32v2, value: u32v4) {
	tex[u32v2(coord)] = value.x;
}

public func imageLoad(tex: RWTexture3D<f32v4>, coord: i32v3) -> f32v4 {
	return tex[u32v3(coord)];
}

public func imageLoad(tex: RWTexture2D<f32v4>, coord: i32v2) -> f32v4 {
	return tex[u32v2(coord)];
}

public func imageLoad(tex: RWTexture2D<u32>, coord: i32v2) -> u32v4 {
	return u32v4(tex[u32v2(coord)], 0u, 0u, 0u);
}

// glsl's mod floors where hlsl's fmod truncates; they differ on negatives
public func mod(x: f32, y: f32) -> f32 {
	return x - y * floor(x / y);
}

public func mod(x: f32v2, y: f32v2) -> f32v2 {
	return x - y * floor(x / y);
}

public func mod(x: f32v3, y: f32v3) -> f32v3 {
	return x - y * floor(x / y);
}

// slang has no matrix inverse of its own; the 4x4 is the general cofactor
// expansion, the 3x3 below the cross-product form
public func inverse(m: f32m4x4) -> f32m4x4 {
	f32m4x4 inv;
	for (u32 r = 0u; r < 4u; ++r) {
		for (u32 c = 0u; c < 4u; ++c) {
			f32m3x3 minorM;
			u32 mr = 0u;
			for (u32 i = 0u; i < 4u; ++i) {
				if (i == c) { continue; }
				u32 mc = 0u;
				for (u32 j = 0u; j < 4u; ++j) {
					if (j == r) { continue; }
					minorM[mr][mc] = m[i][j];
					mc += 1u;
				}
				mr += 1u;
			}
			const f32 minorDet = (
				minorM[0][0] * (minorM[1][1] * minorM[2][2] - minorM[1][2] * minorM[2][1])
				- minorM[0][1] * (minorM[1][0] * minorM[2][2] - minorM[1][2] * minorM[2][0])
				+ minorM[0][2] * (minorM[1][0] * minorM[2][1] - minorM[1][1] * minorM[2][0])
			);
			const f32 sign = ((r + c) % 2u == 0u) ? 1.0f : -1.0f;
			inv[r][c] = sign * minorDet;
		}
	}
	const f32 det = (
		m[0][0] * inv[0][0] + m[0][1] * inv[1][0]
		+ m[0][2] * inv[2][0] + m[0][3] * inv[3][0]
	);
	const f32 invDet = 1.0f / det;
	for (u32 r = 0u; r < 4u; ++r) {
		for (u32 c = 0u; c < 4u; ++c) {
			inv[r][c] *= invDet;
		}
	}
	return inv;
}

// slang has no matrix inverse of its own
public func inverse(m: f32m3x3) -> f32m3x3 {
	const f32v3 c0 = f32v3(m[0][0], m[1][0], m[2][0]);
	const f32v3 c1 = f32v3(m[0][1], m[1][1], m[2][1]);
	const f32v3 c2 = f32v3(m[0][2], m[1][2], m[2][2]);
	const f32v3 r0 = cross(c1, c2);
	const f32v3 r1 = cross(c2, c0);
	const f32v3 r2 = cross(c0, c1);
	const f32 det = dot(c0, r0);
	return f32m3x3(r0 / det, r1 / det, r2 / det);
}

// glsl's atomics return the old value; slang's builtins return it through an
// out parameter, so these keep the glsl call shape. the converter takes the
// address of the target
public func atomicAdd(ptr: u32 *, value: u32) -> u32 {
	Atomic<u32> * slot = (Atomic<u32> *)ptr;
	return slot[0].add(value);
}

public func atomicCompSwap(ptr: u32 *, compare: u32, data: u32) -> u32 {
	Atomic<u32> * slot = (Atomic<u32> *)ptr;
	return slot[0].compareExchange(compare, data);
}
