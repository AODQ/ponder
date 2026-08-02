implementing sample;

import types;

// -----------------------------------------------------------------------------
// -- references
// -----------------------------------------------------------------------------
// [JO20] jarzynski & olano 2020, "hash functions for gpu rendering",
//        jcgt vol 9 no 3. https://jcgt.org/published/0009/03/02/
// -----------------------------------------------------------------------------

// u32 -> [0, 1), from the top 23 bits as a float mantissa
func implUintToUniform(s: u32) -> f32 {
	return asfloat((s >> 9u) | 0x3F800000u) - 1.0f;
}

// pcg4d [JO20]
func implPcg4d(v0: u32v4) -> u32v4 {
	var v: u32v4 = v0 * 1664525u + 1013904223u;
	v.x += v.y * v.w;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v.w += v.y * v.z;
	v ^= v >> 16u;
	v.x += v.y * v.w;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v.w += v.y * v.z;
	return v;
}
