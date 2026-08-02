import "util-random";
import "util-material-openpbr";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u64 sceneVa;
	u32 width;
	u32 height;
	u32 sampleCount;
	u32 kullaContyHandle;
};

[[vk::push_constant]] PC pc;

// first consumer of the FULL ported material chain: lobe selection,
// sampling, pdf and evaluation all go through the public openPbr* api

// per-instance geometry + material record, uploaded by the test. layout
// must match InstanceRecord in test-cornell.cpp (scalar packing)
struct InstanceRecord {
	u64 vertexVa;
	u64 indexVa;
	f32v3 baseColor;
	f32 baseWeight;
	f32 baseMetalness;
	f32 baseDiffuseRoughness;
	f32 specularWeight;
	f32 specularRoughness;
	f32v3 specularColor;
	f32 specularRoughnessAnisotropy;
	f32v3 emission;
	f32 specularIor;
};

static const f32v3 skCameraOrigin = f32v3(0.0, 0.0, -3.0);
static const f32 skTanHalfFov = 0.4;
static const u32 skBounceCount = 8u;

func material_from_record(
	rec: InstanceRecord,
) -> OpenPbrMaterial {
	OpenPbrMaterial mat;
	mat.baseWeight = rec.baseWeight;
	mat.baseColor = rec.baseColor;
	mat.baseMetalness = rec.baseMetalness;
	mat.baseDiffuseRoughness = rec.baseDiffuseRoughness;
	mat.specularWeight = rec.specularWeight;
	mat.specularColor = rec.specularColor;
	mat.specularRoughness = rec.specularRoughness;
	mat.specularRoughnessAnisotropy = rec.specularRoughnessAnisotropy;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularIor = rec.specularIor;
	mat.transmissionWeight = 0.0f;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionScatter = f32v3(0.0f);
	mat.transmissionScatterAnisotropy = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 20.0f;
	mat.subsurfaceWeight = 0.0f;
	mat.subsurfaceColor = f32v3(0.8f);
	mat.subsurfaceRadius = 1.0f;
	mat.subsurfaceRadiusScale = f32v3(1.0f);
	mat.subsurfaceScatterAnisotropy = 0.0f;
	mat.coatWeight = 0.0f;
	mat.coatColor = f32v3(1.0f);
	mat.coatRoughness = 0.0f;
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = 1.6f;
	mat.coatDarkening = 1.0f;
	mat.fuzzWeight = 0.0f;
	mat.fuzzColor = f32v3(1.0f);
	mat.fuzzRoughness = 0.5f;
	mat.emissionLuminance = 0.0f;
	mat.emissionColor = f32v3(0.0f);
	mat.thinFilmWeight = 0.0f;
	mat.thinFilmThickness = 0.5f;
	mat.thinFilmIor = 1.4f;
	mat.geometryOpacity = 1.0f;
	mat.geometryThinWalled = 0.0f;
	return mat;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 px = gl_GlobalInvocationID.x;
	u32 py = gl_GlobalInvocationID.y;
	if (px >= pc.width || py >= pc.height) { return; }
	u32 pixel = py * pc.width + px;

	MaterialTableHandles tables;
	tables.kullaContyEnergy = pc.kullaContyHandle;
	tables.zeltnerLtcParam = 0u;

	InstanceRecord * scene = (InstanceRecord *)(pc.sceneVa);

	// deterministic primary rays; image row 0 is the top (+y in world)
	f32 ndcX = (2.0 * (f32(px) + 0.5) / f32(pc.width) - 1.0);
	f32 ndcY = (2.0 * (f32(py) + 0.5) / f32(pc.height) - 1.0);
	f32v3 primaryDir = normalize(
		f32v3(ndcX * skTanHalfFov, -ndcY * skTanHalfFov, 1.0)
	);

	f32v3 radianceSum = f32v3(0.0);

	for (u32 s = 0u; s < pc.sampleCount; ++s) {
		const u32 h0 = fnPcgHash(pixel * 9781u + s + 1u);
		const u32 h1 = fnPcgHash(h0);
		u64 state = (u64(h0) << 32) | u64(h1);

		f32v3 origin = skCameraOrigin;
		f32v3 dir = primaryDir;
		f32v3 throughput = f32v3(1.0);

		for (u32 bounce = 0u; bounce < skBounceCount; ++bounce) {
			RayQuery<RAY_FLAG_NONE> rq;
			RayDesc rqRay;
			rqRay.Origin = origin;
			rqRay.TMin = // tMin is zero: self-intersection is prevented by the
				// origin offset along the normal; a positive tMin tunnels
				// through adjacent walls at corners
				0.0;
			rqRay.Direction = dir;
			rqRay.TMax = 100.0;
			rq.TraceRayInline(vkofTlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, rqRay);
			while (rq.Proceed()) { }

			bool hit = (
				rq.CommittedStatus()
				!= COMMITTED_NOTHING
			);
			if (!hit) { break; }

			f32 t = rq.CommittedRayT();
			u32 inst = u32(rq.CommittedInstanceID());
			u32 prim = u32(rq.CommittedPrimitiveIndex());
			f32m4x3 objectToWorld = rq.CommittedObjectToWorld3x4();

			InstanceRecord rec = scene[inst];

			// geometric normal from the hit triangle, transformed by the
			// instance rotation (transforms are rotation + translation
			// only; scale is baked into the blas vertices)
			u32 * indices = (u32 *)(rec.indexVa);
			f32v3 * verts = (f32v3 *)(rec.vertexVa);
			f32v3 p0 = verts[indices[3u * prim + 0u]];
			f32v3 p1 = verts[indices[3u * prim + 1u]];
			f32v3 p2 = verts[indices[3u * prim + 2u]];
			f32v3 geomN = normalize(
				mul(f32m3x3(objectToWorld), cross(p1 - p0, p2 - p0)));
			f32v3 n = geomN;
			if (dot(n, dir) > 0.0) { n = -n; }

			// one-sided emitter: only the winding front face emits (the
			// light quad is wound facing down into the box)
			if (dot(geomN, dir) < 0.0) {
				radianceSum += throughput * rec.emission;
			}

			OpenPbrMaterial mat = material_from_record(rec);
			ShadingFrame frame = shadingFrameFromNormal(n);
			f32v3 wi = -dir;

			f32 pdf;
			bool sampledTransmission;
			bool sampledSubsurfaceEntry;
			u32 pickedLobe;
			f32v3 wo = openPbrSampleWo(
				tables,
				frame,
				frame,
				wi,
				mat,
				pdf,
				sampledTransmission,
				sampledSubsurfaceEntry,
				pickedLobe,
				state,
				/*isInsideMedium=*/false,
				/*dispersedIor=*/mat.specularIor
			);
			if (pdf <= 0.0 || dot(n, wo) <= 0.0) { break; }

			f32v3 f = openPbrEvaluateF(
				tables,
				frame,
				frame,
				wi,
				wo,
				mat,
				/*isInsideMedium=*/false,
				/*dispersedIor=*/mat.specularIor
			);
			throughput *= f * dot(n, wo) / pdf;

			f32v3 hitPos = origin + t * dir;
			dir = wo;
			origin = hitPos + n * 0.001;
		}
	}

	f32v3 radiance = radianceSum / f32(pc.sampleCount);
	u32 pixelCount = pc.width * pc.height;
	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[pixel] = radiance.r;
	outBuf[pixelCount + pixel] = radiance.g;
	outBuf[2u * pixelCount + pixel] = radiance.b;
}
