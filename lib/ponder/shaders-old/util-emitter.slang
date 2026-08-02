module "util-emitter";
__exported import "util-types";
__exported import "util-emitter-shared";
__exported import "util-resolve-pc";
// next-event-estimation sampling over emissive-mesh triangles. the emitter
// list is one entry per emissive meshlet (see emitter-shared.h); a pick
// chooses an entry by power, then a triangle uniformly within it, then a
// point on that triangle.
//
// the includer must already have pulled in mor-shared.h and resolve_pc.h --
// this reads triangles straight out of GpuResolveModelIndirect



// (TODO REVIEW)

// own name rather than PI: util-material-openpbr.glsl defines PI
// unconditionally, so a shared spelling would redefine depending on include
// order
public static const f32 skEmitterPi = 3.14159265358979323846f;

// -----------------------------------------------------------------------------
// -- triangle fetch
// -----------------------------------------------------------------------------

public struct EmitterTriangle {
	public f32v3 p0;
	public f32v3 p1;
	public f32v3 p2;
};

// world-space corners of a triangle in the flat primitiveIndex numbering ray
// hits report, so forward sampling and the backward hit lookup address the
// same triangle
public func utilEmitterTriangle(
	model: GpuResolveModelIndirect,
	primitiveIndex: u32,
) -> EmitterTriangle {
	u32 * flatIdxBuf = (u32 *)(model.flatIndices);
	const u32 i0 = flatIdxBuf[primitiveIndex * 3u + 0u];
	const u32 i1 = flatIdxBuf[primitiveIndex * 3u + 1u];
	const u32 i2 = flatIdxBuf[primitiveIndex * 3u + 2u];

	f32v3 * positionBuf = (f32v3 *)(model.positions);
	EmitterTriangle tri;
	tri.p0 = (mul(model.modelMatrix, f32v4(positionBuf[i0], 1.0f))).xyz;
	tri.p1 = (mul(model.modelMatrix, f32v4(positionBuf[i1], 1.0f))).xyz;
	tri.p2 = (mul(model.modelMatrix, f32v4(positionBuf[i2], 1.0f))).xyz;
	return tri;
}

public struct EmitterTriangleGeometry {
	public f32 area;
	public f32v3 normal;
};

// area and normal of the *transformed* triangle; taking the cross product
// after the transform is what makes a non-uniformly scaled instance's area
// exact, which the cpu-side power estimate only approximates
public func utilEmitterTriangleGeometry(
	tri: EmitterTriangle,
) -> EmitterTriangleGeometry {
	const f32v3 crossWorld = cross(tri.p1 - tri.p0, tri.p2 - tri.p0);
	const f32 lenCross = length(crossWorld);
	EmitterTriangleGeometry geo;
	geo.area = 0.5f * lenCross;
	geo.normal = lenCross > 0.0f ? crossWorld / lenCross : f32v3(0.0f);
	return geo;
}

// -----------------------------------------------------------------------------
// -- emitter pick (power cdf binary search)
// -----------------------------------------------------------------------------

public struct EmitterPick {
	public bool valid;
	public u32 lightIndex;
	// probability mass of picking this entry, i.e. power / totalPower
	public f32 pickPdf;
};

public func utilEmitterPick(
	lights: GpuEmitterLight *,
	lightCount: u32,
	u: f32,
) -> EmitterPick {
	EmitterPick pick;
	pick.valid = lightCount > 0u;
	pick.lightIndex = 0u;
	pick.pickPdf = 0.0f;
	if (!pick.valid) {
		return pick;
	}

	u32 lo = 0u;
	u32 hi = lightCount - 1u;
	while (lo < hi) {
		const u32 mid = (lo + hi) / 2u;
		if (lights[mid].cdf < u) {
			lo = mid + 1u;
		} else {
			hi = mid;
		}
	}

	const f32 prevCdf = lo == 0u ? 0.0f : lights[lo - 1u].cdf;
	pick.lightIndex = lo;
	pick.pickPdf = lights[lo].cdf - prevCdf;
	pick.valid = pick.pickPdf > 0.0f;
	return pick;
}

// pick probability of an already-known entry; the backward (bsdf-hit) mis
// weight found its emitter by a ray hit rather than by utilEmitterPick
public func utilEmitterPickPdf(
	lights: GpuEmitterLight *,
	lightIndex: u32,
) -> f32 {
	const f32 prevCdf = (
		lightIndex == 0u ? 0.0f : lights[lightIndex - 1u].cdf
	);
	return lights[lightIndex].cdf - prevCdf;
}

// -----------------------------------------------------------------------------
// -- spherical triangle solid angle + sampling (arvo 1995)
// -----------------------------------------------------------------------------

// numerically stable angle between two unit vectors; the naive acos(dot)
// loses most of its precision as the angle approaches 0 or pi, which is
// exactly the regime the spherical-triangle angles live in for small emitters
public func utilEmitterAngleBetween(
	a: f32v3,
	b: f32v3,
) -> f32 {
	if (dot(a, b) < 0.0f) {
		return skEmitterPi - 2.0f * asin(clamp(length(a + b) * 0.5f, 0.0f, 1.0f));
	}
	return 2.0f * asin(clamp(length(b - a) * 0.5f, 0.0f, 1.0f));
}

public func utilEmitterGramSchmidt(
	v: f32v3,
	w: f32v3,
) -> f32v3 {
	return v - dot(v, w) * w;
}

// interior angles of the spherical triangle the emitter subtends at `from`.
// the solid angle is their excess over pi; sampling needs the angles
// themselves, and the pdf needs the excess, so both come from here and can
// never disagree about which triangle is which size
public struct EmitterSphericalTriangle {
	public bool valid;
	// unit directions to the three corners
	public f32v3 a;
	public f32v3 b;
	public f32v3 c;
	public f32 alpha;
	// alpha + beta + gamma; the solid angle is this minus pi
	public f32 anglesSum;
	public f32 solidAngle;
};

public func utilEmitterSphericalTriangle(
	tri: EmitterTriangle,
	from: f32v3,
) -> EmitterSphericalTriangle {
	EmitterSphericalTriangle st;
	st.valid = false;
	st.a = f32v3(0.0f);
	st.b = f32v3(0.0f);
	st.c = f32v3(0.0f);
	st.alpha = 0.0f;
	st.anglesSum = 0.0f;
	st.solidAngle = 0.0f;

	const f32v3 da = tri.p0 - from;
	const f32v3 db = tri.p1 - from;
	const f32v3 dc = tri.p2 - from;
	if (
		dot(da, da) <= 0.0f || dot(db, db) <= 0.0f || dot(dc, dc) <= 0.0f
	) {
		return st;
	}
	st.a = normalize(da);
	st.b = normalize(db);
	st.c = normalize(dc);

	f32v3 nAb = cross(st.a, st.b);
	f32v3 nBc = cross(st.b, st.c);
	f32v3 nCa = cross(st.c, st.a);
	if (
		dot(nAb, nAb) <= 0.0f || dot(nBc, nBc) <= 0.0f || dot(nCa, nCa) <= 0.0f
	) {
		return st;
	}
	nAb = normalize(nAb);
	nBc = normalize(nBc);
	nCa = normalize(nCa);

	st.alpha = utilEmitterAngleBetween(nAb, -nCa);
	const f32 beta = utilEmitterAngleBetween(nBc, -nAb);
	const f32 gamma = utilEmitterAngleBetween(nCa, -nBc);
	st.anglesSum = st.alpha + beta + gamma;
	st.solidAngle = st.anglesSum - skEmitterPi;
	st.valid = st.solidAngle > 0.0f;
	return st;
}

// pbrt-v4's stability window for spherical sampling: below it the spherical
// excess is lost to float cancellation, above it the arc reconstruction is.
// area sampling covers both tails
#define EMITTER_SOLID_ANGLE_MIN 3e-4f
public static const f32 EMITTER_SOLID_ANGLE_MAX = 6.22f;
public func utilEmitterUseSphericalSampling(
	st: EmitterSphericalTriangle,
) -> bool {
	return (
		st.valid
		&& st.solidAngle > EMITTER_SOLID_ANGLE_MIN
		&& st.solidAngle < EMITTER_SOLID_ANGLE_MAX
	);
}

// arvo 1995 \S4: samples the spherical triangle uniformly in solid angle and
// returns the barycentrics of the corresponding point on the flat triangle
public func utilEmitterSampleSphericalTriangleBarycentric(
	tri: EmitterTriangle,
	st: EmitterSphericalTriangle,
	from: f32v3,
	xi: f32v2,
) -> f32v3 {
	// uniformly sample the sub-triangle area, carried as area + pi
	const f32 apPi = lerp(skEmitterPi, st.anglesSum, xi.x);

	// cos of the angle at the sampled sub-triangle's third vertex
	const f32 cosAlpha = cos(st.alpha);
	const f32 sinAlpha = sin(st.alpha);
	const f32 sinPhi = sin(apPi) * cosAlpha - cos(apPi) * sinAlpha;
	const f32 cosPhi = cos(apPi) * cosAlpha + sin(apPi) * sinAlpha;
	const f32 k1 = cosPhi + cosAlpha;
	const f32 k2 = sinPhi - sinAlpha * dot(st.a, st.b);
	const f32 denom = (k2 * sinPhi + k1 * cosPhi) * sinAlpha;
	const f32 cosBp = (
		denom != 0.0f
			? clamp(
				(k2 + (k2 * cosPhi - k1 * sinPhi) * cosAlpha) / denom,
				-1.0f, 1.0f
			)
			: 1.0f
	);

	// c' along the arc between b' and a
	const f32 sinBp = sqrt(max(1.0f - cosBp * cosBp, 0.0f));
	const f32v3 caPerp = utilEmitterGramSchmidt(st.c, st.a);
	const f32v3 cp = (
		dot(caPerp, caPerp) > 0.0f
			? cosBp * st.a + sinBp * normalize(caPerp)
			: st.a
	);

	// sample the arc from b toward c'
	const f32 cosTheta = 1.0f - xi.y * (1.0f - dot(cp, st.b));
	const f32 sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));
	const f32v3 cpPerp = utilEmitterGramSchmidt(cp, st.b);
	const f32v3 w = (
		dot(cpPerp, cpPerp) > 0.0f
			? cosTheta * st.b + sinTheta * normalize(cpPerp)
			: st.b
	);

	// intersect the sampled direction with the triangle plane for barycentrics
	const f32v3 e1 = tri.p1 - tri.p0;
	const f32v3 e2 = tri.p2 - tri.p0;
	const f32v3 s1 = cross(w, e2);
	const f32 divisor = dot(s1, e1);
	if (divisor == 0.0f) {
		return f32v3(1.0f / 3.0f);
	}
	const f32 invDivisor = 1.0f / divisor;
	const f32v3 s = from - tri.p0;
	f32 b1 = clamp(dot(s, s1) * invDivisor, 0.0f, 1.0f);
	f32 b2 = clamp(dot(w, cross(s, e1)) * invDivisor, 0.0f, 1.0f);
	const f32 bSum = b1 + b2;
	if (bSum > 1.0f) {
		b1 /= bSum;
		b2 /= bSum;
	}
	return f32v3(1.0f - b1 - b2, b1, b2);
}

// -----------------------------------------------------------------------------
// -- point sampling on a picked emitter
// -----------------------------------------------------------------------------

public struct EmitterSample {
	public bool valid;
	public u32 lightIndex;
	public u32 primitiveIndex;
	public f32v3 position;
	// geometric normal of the sampled triangle
	public f32v3 normal;
	public f32v3 direction;
	public f32 distance;
	// barycentrics of the sampled point, for evaluating the emitter's
	// material at the sample rather than trusting the precomputed average
	public f32v3 barycentric;
	// solid-angle density, including the emitter pick and the uniform
	// triangle pick within that emitter
	public f32 pdf;
};

// solid-angle density of a point on a known triangle as seen from `from`.
// two-sided: abs() on the light cosine, matching the bsdf-hit branch, which
// adds emission with no facing test
public func utilEmitterPointPdf(
	tri: EmitterTriangle,
	from: f32v3,
	direction: f32v3,
	distance: f32,
) -> f32 {
	const EmitterSphericalTriangle st = (
		utilEmitterSphericalTriangle(tri, from)
	);
	if (utilEmitterUseSphericalSampling(st)) {
		// uniform in solid angle, so no distance/cosine jacobian at all
		return 1.0f / st.solidAngle;
	}
	const EmitterTriangleGeometry geo = utilEmitterTriangleGeometry(tri);
	const f32 cosAtLight = abs(dot(geo.normal, -direction));
	if (geo.area <= 0.0f || cosAtLight <= 0.0f || distance <= 0.0f) {
		return 0.0f;
	}
	return (distance * distance) / (cosAtLight * geo.area);
}

// full solid-angle density of the emitter strategy for a known triangle;
// the backward (bsdf-hit) mis weight's other half
public func utilEmitterPdf(
	lights: GpuEmitterLight *,
	lightIndex: u32,
	model: GpuResolveModelIndirect,
	primitiveIndex: u32,
	from: f32v3,
	direction: f32v3,
	distance: f32,
) -> f32 {
	const GpuEmitterLight light = lights[lightIndex];
	if (light.kind != EMITTER_KIND_MESH || light.triangleCount == 0u) {
		return 0.0f;
	}
	const EmitterTriangle tri = utilEmitterTriangle(model, primitiveIndex);
	const f32 pointPdf = (
		utilEmitterPointPdf(tri, from, direction, distance)
	);
	if (pointPdf <= 0.0f) {
		return 0.0f;
	}
	const f32 pickPdf = utilEmitterPickPdf(lights, lightIndex);
	return pickPdf * pointPdf / f32(light.triangleCount);
}

// samples a point on an *already picked* mesh entry. split out of
// utilEmitterSample so a caller whose list also holds non-mesh entries (the
// environment map) can branch on the picked entry's kind before committing to
// a triangle, without spending a second pick draw.
//
// xi3: (triangle pick, barycentric u, barycentric v). the triangle pick gets
// its own draw rather than reusing the pick draw's position inside its cdf
// interval -- that remainder is uniform in theory, but with a few thousand
// emitters pickPdf approaches float resolution and the rescaled value
// collapses to a handful of distinct triangles
public func utilEmitterSampleFromPick(
	lights: GpuEmitterLight *,
	pick: EmitterPick,
	models: GpuResolveModelIndirect *,
	from: f32v3,
	xi3: f32v3,
) -> EmitterSample {
	EmitterSample result;
	result.valid = false;
	result.lightIndex = 0u;
	result.primitiveIndex = 0u;
	result.position = f32v3(0.0f);
	result.normal = f32v3(0.0f);
	result.direction = f32v3(0.0f);
	result.distance = 0.0f;
	result.barycentric = f32v3(0.0f);
	result.pdf = 0.0f;

	if (!pick.valid) {
		return result;
	}
	const GpuEmitterLight light = lights[pick.lightIndex];
	if (light.kind != EMITTER_KIND_MESH || light.triangleCount == 0u) {
		return result;
	}

	const f32v4 xi4 = f32v4(0.0f, xi3.x, xi3.y, xi3.z);
	const u32 triLocal = (
		min(u32(xi4.y * f32(light.triangleCount)), light.triangleCount - 1u)
	);
	result.primitiveIndex = light.primitiveIndexBase + triLocal;
	result.lightIndex = pick.lightIndex;

	const GpuResolveModelIndirect model = models[light.modelDrawIndex];
	const EmitterTriangle tri = utilEmitterTriangle(model, result.primitiveIndex);
	const EmitterTriangleGeometry geo = utilEmitterTriangleGeometry(tri);
	if (geo.area <= 0.0f) {
		return result;
	}

	const EmitterSphericalTriangle st = utilEmitterSphericalTriangle(tri, from);
	f32 pointPdf;
	if (utilEmitterUseSphericalSampling(st)) {
		result.barycentric = (
			utilEmitterSampleSphericalTriangleBarycentric(
				tri, st, from, f32v2(xi4.z, xi4.w)
			)
		);
		pointPdf = 1.0f / st.solidAngle;
	} else {
		// uniform area sampling; the sqrt warp maps the unit square onto the
		// triangle without distorting density
		const f32 su0 = sqrt(xi4.z);
		const f32 b1 = 1.0f - su0;
		const f32 b2 = xi4.w * su0;
		result.barycentric = f32v3(1.0f - b1 - b2, b1, b2);
	}

	result.position = (
		tri.p0 * result.barycentric.x
		+ tri.p1 * result.barycentric.y
		+ tri.p2 * result.barycentric.z
	);
	result.normal = geo.normal;

	const f32v3 toLight = result.position - from;
	const f32 dist2 = dot(toLight, toLight);
	if (dist2 <= 0.0f) {
		return result;
	}
	result.distance = sqrt(dist2);
	result.direction = toLight / result.distance;

	if (!utilEmitterUseSphericalSampling(st)) {
		const f32 cosAtLight = abs(dot(geo.normal, -result.direction));
		if (cosAtLight <= 0.0f) {
			return result;
		}
		pointPdf = dist2 / (cosAtLight * geo.area);
	}

	result.pdf = pick.pickPdf * pointPdf / f32(light.triangleCount);
	result.valid = result.pdf > 0.0f;
	return result;
}

// pick + sample in one call, for a list known to hold mesh entries only.
// xi4: (emitter pick, triangle pick, barycentric u, barycentric v)
public func utilEmitterSample(
	lights: GpuEmitterLight *,
	lightCount: u32,
	models: GpuResolveModelIndirect *,
	from: f32v3,
	xi4: f32v4,
) -> EmitterSample {
	const EmitterPick pick = utilEmitterPick(lights, lightCount, xi4.x);
	return (
		utilEmitterSampleFromPick(
			lights, pick, models, from, f32v3(xi4.y, xi4.z, xi4.w)
		)
	);
}

// -----------------------------------------------------------------------------
// -- backward lookup (a bsdf-sampled ray happened to hit an emissive triangle)
// -----------------------------------------------------------------------------

public struct EmitterHitLookup {
	public bool found;
	public u32 lightIndex;
};

// scans only this model's own emitters; a miss means the triangle is not
// sampleable, so its mis weight must stay 1 rather than blend against 0
public func utilEmitterFindForHit(
	lights: GpuEmitterLight *,
	emitterLightOffset: u32,
	emitterLightCount: u32,
	primitiveIndex: u32,
) -> EmitterHitLookup {
	EmitterHitLookup result;
	result.found = false;
	result.lightIndex = 0u;
	for (u32 i = 0u; i < emitterLightCount; ++i) {
		const u32 idx = emitterLightOffset + i;
		const GpuEmitterLight light = lights[idx];
		if (
			light.kind == EMITTER_KIND_MESH
			&& light.triangleCount > 0u
			&& primitiveIndex >= light.primitiveIndexBase
			&& primitiveIndex < light.primitiveIndexBase + light.triangleCount
		) {
			result.found = true;
			result.lightIndex = idx;
			return result;
		}
	}
	return result;
}

// (TODO REVIEW)

