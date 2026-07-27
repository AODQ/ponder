#ifndef UTIL_SHADING_FRAME_REFERENCE_GLSL
#define UTIL_SHADING_FRAME_REFERENCE_GLSL

#ifndef f32
#define i32v2 ivec2
#define f32 float
#define f32v3 vec3
#endif

// independent cross-check only, not used by the real ponder implementation.
// ported directly from the paper itself (not materialx this time): Duff,
// Burgess, Christensen, Hery, Kensler, Liani, Villemin, "Building an
// Orthonormal Basis, Revisited", JCGT vol 6 no 1, 2017, Listing 3
// ("branchlessONB"). GLSL has no copysignf, so the sign is computed the
// same way the paper itself suggests as an equally-valid alternative
// ("sign = n.z>=0.0f ? 1.0f : -1.0f").
void branchlessONB(const f32v3 n, out f32v3 b1, out f32v3 b2)
{
	f32 sign = n.z >= 0.0f ? 1.0f : -1.0f;
	const f32 a = -1.0f / (sign + n.z);
	const f32 b = n.x * n.y * a;
	b1 = f32v3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
	b2 = f32v3(b, sign + n.y * n.y * a, -n.y);
}

#endif
