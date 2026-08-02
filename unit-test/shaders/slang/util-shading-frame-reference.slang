#pragma once
__exported import "util-types";


// independent cross-check only, not used by the real ponder implementation.
// ported directly from the paper itself (not materialx this time): Duff,
// Burgess, Christensen, Hery, Kensler, Liani, Villemin, "Building an
// Orthonormal Basis, Revisited", JCGT vol 6 no 1, 2017, Listing 3
// ("branchlessONB"). GLSL has no copysignf, so the sign is computed the
// same way the paper itself suggests as an equally-valid alternative
// ("sign = n.z>=0.0f ? 1.0f : -1.0f").
public func branchlessONB(
	n: f32v3,
	out b1: f32v3,
	out b2: f32v3,
)
{
	f32 sign = n.z >= 0.0f ? 1.0f : -1.0f;
	const f32 a = -1.0f / (sign + n.z);
	const f32 b = n.x * n.y * a;
	b1 = f32v3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
	b2 = f32v3(b, sign + n.y * n.y * a, -n.y);
}

