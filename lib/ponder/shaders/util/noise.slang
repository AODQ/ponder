implementing util;

namespace vkof {

public struct NoiseSample {
	public f32 value;
	public f32v3 gradient;
};

// value noise in [-1, 1] and its analytic gradient
public func noiseValueGradient(p: f32v3) -> NoiseSample {
	let cell: f32v3 = floor(p);
	let f: f32v3 = p - cell;
	// smoothstep weights and their derivatives
	let w: f32v3 = f * f * (3.0f - 2.0f * f);
	let dw: f32v3 = 6.0f * f * (1.0f - f);
	let base: u32v3 = u32v3(i32v3(cell));
	// corner i sits at offset (i & 1, (i >> 1) & 1, i >> 2)
	var c: f32[8];
	[ForceUnroll]
	for (u32 i = 0u; i < 8u; ++i) {
		let offset: u32v3 = u32v3(i & 1u, (i >> 1u) & 1u, i >> 2u);
		c[i] = f32(implNoiseHashPcg3d(base + offset).x) * (1.0f / 4294967296.0f);
	}
	// trilinear interpolation expanded into polynomial terms
	let k0: f32 = c[0];
	let k1: f32 = c[1] - c[0];
	let k2: f32 = c[2] - c[0];
	let k3: f32 = c[4] - c[0];
	let k4: f32 = c[0] - c[1] - c[2] + c[3];
	let k5: f32 = c[0] - c[2] - c[4] + c[6];
	let k6: f32 = c[0] - c[1] - c[4] + c[5];
	let k7: f32 = (
		-c[0] + c[1] + c[2] - c[3] + c[4] - c[5] - c[6] + c[7]
	);
	let value: f32 = (
		k0 + k1 * w.x + k2 * w.y + k3 * w.z
		+ k4 * w.x * w.y + k5 * w.y * w.z + k6 * w.z * w.x
		+ k7 * w.x * w.y * w.z
	);
	let gradient: f32v3 = (
		dw
		* f32v3(
			k1 + k4 * w.y + k6 * w.z + k7 * w.y * w.z,
			k2 + k5 * w.z + k4 * w.x + k7 * w.z * w.x,
			k3 + k6 * w.x + k5 * w.y + k7 * w.x * w.y
		)
	);
	return { 2.0f * value - 1.0f, 2.0f * gradient };
}

// value noise in [-1, 1], cheaper than noiseValueGradient(p).value
public func noiseValue(p: f32v3) -> f32 {
	let cell: f32v3 = floor(p);
	let w: f32v3 = smoothstep(0.0f, 1.0f, p - cell);
	let base: u32v3 = u32v3(i32v3(cell));
	var c: f32[8];
	[ForceUnroll]
	for (u32 i = 0u; i < 8u; ++i) {
		let offset: u32v3 = u32v3(i & 1u, (i >> 1u) & 1u, i >> 2u);
		c[i] = f32(implNoiseHashPcg3d(base + offset).x) * (1.0f / 4294967296.0f);
	}
	let value: f32 = (
		lerp(
			lerp(lerp(c[0], c[1], w.x), lerp(c[2], c[3], w.x), w.y),
			lerp(lerp(c[4], c[5], w.x), lerp(c[6], c[7], w.x), w.y),
			w.z
		)
	);
	return 2.0f * value - 1.0f;
}

// fractal brownian motion in [-1, 1]
public func noiseFbm<u32 Octaves = 4>(p: f32v3) -> f32 {
	var sum: f32 = 0.0f;
	var amplitude: f32 = 0.5f;
	var q: f32v3 = p;
	[ForceUnroll]
	for (u32 octave = 0u; octave < Octaves; ++octave) {
		sum += amplitude * noiseValue(q);
		q *= 2.0f;
		amplitude *= 0.5f;
	}
	return sum / 0.9375f;
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

// pcg3d, jarzynski & olano 2020
func implNoiseHashPcg3d(v: u32v3) -> u32v3 {
	var u: u32v3 = v * 1664525u + 1013904223u;
	u.x += u.y * u.z;
	u.y += u.z * u.x;
	u.z += u.x * u.y;
	u ^= u >> 16u;
	u.x += u.y * u.z;
	u.y += u.z * u.x;
	u.z += u.x * u.y;
	return u;
}
