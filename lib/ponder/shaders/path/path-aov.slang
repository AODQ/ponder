implementing path;

// TODO
// this is a bit of slop to get DLSS to work, probably want to clean ti up

import material;
import model;
import types;
import util;

namespace vkof {

// AOV surface data
public struct PathAov {
	// w is 1 on a surface, 0 otherwise
	public f32v4 position;
	public f32v3 shadingNormal;
	// linear rec.709; 1 off a surface
	public f32v3 albedo;
};

public func pathAovFromHit(hit: TraceTransportHit) -> PathAov {
	var aov: PathAov;
	aov.position = f32v4(0.0f);
	aov.shadingNormal = f32v3(0.0f);
	aov.albedo = f32v3(1.0f);
	if (hit.event != TransportEvent::Surface) { return aov; }

	// -- geometry, facing the incoming ray
	var geometry: SurfaceGeometry = (
		surfaceGeometryFromHit(
			/*hit=*/ hit.surface.hit,
			/*worldPosition=*/ hit.surface.position
		)
	);
	if (dot(geometry.geometricNormal, hit.incomingDirection) > 0.0f) {
		geometry.shadingNormal = -geometry.shadingNormal;
	}

	// -- material
	let mtrl: MaterialRef = materialRefFromHit(hit.surface.hit);
	aov.position = f32v4(hit.position, 1.0f);
	aov.shadingNormal = geometry.shadingNormal;
	aov.albedo = MaterialBackend_Diffuse.resolve(mtrl, geometry).albedo.rgb;
	return aov;
}

// DLSS ray reconstruction guides; off a surface they hold the guide's sky
// defaults
public struct PathRrGuide {
	// linear rec.709
	public f32v3 diffuseAlbedo;
	// directional specular albedo, linear rec.709
	public f32v3 specularAlbedo;
	// linear (perceptual) roughness
	public f32 roughness;
	// world distance along the mirror reflection to the next surface
	public f32 specularHitDistance;
	// the surface depth, motion and normals follow: the primary hit, or what
	// smooth glass shows. w is 1 on a surface, 0 for the sky
	public f32v4 position;
	public f32v3 normal;
};

// the guide's value for the sky and for mirror rays that escape
static let skRrSkyDistance: f32 = 65504.0f;
// glass the guides look through: smooth enough to show a sharp background
static let skRrGlassRoughnessMax: f32 = 0.3f;
// glass surfaces followed before the guides stop at the last one
static let skRrGlassChainMax: u32 = 4u;

func implRrSky() -> PathRrGuide {
	var guide: PathRrGuide;
	guide.diffuseAlbedo = f32v3(0.5f);
	guide.specularAlbedo = f32v3(0.0f);
	guide.roughness = 0.0f;
	guide.specularHitDistance = skRrSkyDistance;
	guide.position = f32v4(0.0f);
	guide.normal = f32v3(0.0f);
	return guide;
}

// one surface's guides, before the hit distance
struct ImplRrSurface {
	PathRrGuide guide;
	f32v3 mirror;
	// fraction of the view that passes through the surface along refracted
	f32 transmission;
	f32v3 refracted;
};

// the hit distance along a surface's mirror direction
func implRrHitDistance(guide: PathRrGuide, mirror: f32v3) -> f32 {
	if (guide.position.w <= 0.0f) { return skRrSkyDistance; }
	let mirrorHit: ImplTraceStepHit = (
		implTraceStep(
			/*origin=*/ guide.position.xyz,
			/*dir=*/ mirror,
			/*distMin=*/ rayEpsilon(guide.position.xyz),
			/*distMax=*/ skEnvironmentDistance
		)
	);
	return mirrorHit.hit() ? mirrorHit.distance : skRrSkyDistance;
}

public func pathRrGuideFromHit(hit: TraceTransportHit) -> PathRrGuide {
	if (hit.event != TransportEvent::Surface) { return implRrSky(); }
	let first: ImplRrSurface = implRrSurface(hit);
	var guide: PathRrGuide = first.guide;
	var mirror: f32v3 = first.mirror;

	// -- smooth glass: follow the refraction to the surface it shows
	if (first.transmission > 0.0f) {
		var through: PathRrGuide = implRrSky();
		var throughMirror: f32v3 = f32v3(0.0f);
		var origin: f32v3 = hit.position;
		var direction: f32v3 = first.refracted;
		for (u32 i = 0u; i < skRrGlassChainMax; ++i) {
			let step: ImplTraceStepHit = (
				implTraceStep(
					/*origin=*/ origin,
					/*dir=*/ direction,
					/*distMin=*/ rayEpsilon(origin),
					/*distMax=*/ skEnvironmentDistance
				)
			);
			if (!step.hit()) {
				through = implRrSky();
				break;
			}
			let next: ImplRrSurface = (
				implRrSurface(
					implTransportHitFromStep(
						/*stepHit=*/ step,
						/*chordOri=*/ origin,
						/*chordDirection=*/ direction,
						/*elapsedBefore=*/ 0.0f
					)
				)
			);
			through = next.guide;
			throughMirror = next.mirror;
			if (next.transmission <= 0.0f) { break; }
			origin = next.guide.position.xyz;
			direction = next.refracted;
		}
		// albedos and roughness blend; depth, motion, normals and the hit
		// distance must agree, so they follow whichever surface dominates
		let t: f32 = first.transmission;
		guide.diffuseAlbedo = lerp(guide.diffuseAlbedo, through.diffuseAlbedo, t);
		guide.specularAlbedo = (
			lerp(guide.specularAlbedo, through.specularAlbedo, t)
		);
		guide.roughness = lerp(guide.roughness, through.roughness, t);
		if (t > 0.5f) {
			guide.position = through.position;
			guide.normal = through.normal;
			mirror = throughMirror;
		}
	}
	guide.specularHitDistance = implRrHitDistance(guide, mirror);
	return guide;
}

func implRrSurface(hit: TraceTransportHit) -> ImplRrSurface {
	var surface: ImplRrSurface;
	var guide: PathRrGuide = implRrSky();

	// -- geometry, facing the incoming ray
	var geometry: SurfaceGeometry = (
		surfaceGeometryFromHit(
			/*hit=*/ hit.surface.hit,
			/*worldPosition=*/ hit.surface.position
		)
	);
	if (dot(geometry.geometricNormal, hit.incomingDirection) > 0.0f) {
		geometry.shadingNormal = -geometry.shadingNormal;
	}

	// -- material, split into a diffuse base and a specular f0
	// guides are rgb: only the scalars and rgb inputs of params are read, so
	// any wavelengths do
	let mtrl: MaterialRef = materialRefFromHit(hit.surface.hit);
	let params: MaterialBackend_OpenPbr.Params = (
		MaterialBackend_OpenPbr.resolve(
			mtrl, geometry, wavelengthSampleUniform(0.5f)
		)
	);
	let baseColor: f32v3 = (
		materialComponentEvaluate(
			/*mtrl=*/ mtrl,
			/*component=*/ &mtrl.material.baseColor,
			/*uv=*/ geometry.uv,
			/*lod=*/ geometry.textureLod
		)
	);
	let specularColor: f32v3 = (
		materialComponentEvaluate(
			/*mtrl=*/ mtrl,
			/*component=*/ &mtrl.material.specularColor,
			/*uv=*/ geometry.uv,
			/*lod=*/ geometry.textureLod
		)
	);
	let base: f32v3 = baseColor * params.baseWeight;
	let dielectric: f32 = 1.0f - params.baseMetalness;
	let iorRatio: f32 = (
		(params.specularIor - 1.0f) / (params.specularIor + 1.0f)
	);
	let iorF0: f32 = iorRatio * iorRatio;
	let f0: f32v3 = (
		lerp(
			saturate(iorF0 * specularColor * params.specularWeight),
			base,
			params.baseMetalness
		)
	);
	let nDotV: f32 = dot(geometry.shadingNormal, -hit.incomingDirection);
	guide.diffuseAlbedo = base * dielectric * (1.0f - params.transmissionWeight);
	guide.specularAlbedo = (
		implEnvBrdfApprox2(
			/*specularColor=*/ f0,
			/*alpha=*/ params.specularRoughness * params.specularRoughness,
			/*nDotV=*/ nDotV
		)
	);
	guide.roughness = params.specularRoughness;
	guide.position = f32v4(hit.position, 1.0f);
	guide.normal = geometry.shadingNormal;
	surface.guide = guide;

	// -- mirror direction for the hit distance: a ray of its own, since the
	// path picks one lobe and may not have reflected. a normal map can tip
	// the mirror below the surface; reflect about the geometric normal then
	let entering: bool = dot(geometry.geometricNormal, hit.incomingDirection) < 0.0f;
	let facingGeometric: f32v3 = (
		entering ? geometry.geometricNormal : -geometry.geometricNormal
	);
	surface.mirror = reflect(hit.incomingDirection, geometry.shadingNormal);
	if (dot(surface.mirror, facingGeometric) <= 0.0f) {
		surface.mirror = reflect(hit.incomingDirection, facingGeometric);
	}

	// -- view through smooth glass: unpolarized schlick fresnel off the ior
	surface.transmission = 0.0f;
	surface.refracted = f32v3(0.0f);
	if (
		params.transmissionWeight > 0.0f
		&& params.specularRoughness <= skRrGlassRoughnessMax
	) {
		let thinWalled: bool = params.geometryThinWalled > 0.5f;
		let eta: f32 = entering ? 1.0f / params.specularIor : params.specularIor;
		surface.refracted = (
			thinWalled
			? hit.incomingDirection
			: refract(hit.incomingDirection, geometry.shadingNormal, eta)
		);
		// total internal reflection shows nothing behind
		if (dot(surface.refracted, surface.refracted) > 0.0f) {
			let fresnel: f32 = (
				iorF0 + (1.0f - iorF0) * pow(1.0f - saturate(nDotV), 5.0f)
			);
			surface.transmission = (
				params.transmissionWeight * dielectric * (1.0f - fresnel)
			);
		}
	}
	return surface;
}

} // namespace vkof

// the DLSS-RR integration guide's appendix (ray tracing gems, chapter 32)
func implEnvBrdfApprox2(specularColor: f32v3, alpha: f32, nDotV: f32) -> f32v3 {
	let nv: f32 = abs(nDotV);
	let x: f32v4 = f32v4(1.0f, nv, nv * nv, nv * nv * nv);
	let y: f32v4 = f32v4(1.0f, alpha, alpha * alpha, alpha * alpha * alpha);
	let m1: float2x2 = float2x2(0.99044f, -1.28514f, 1.29678f, -0.755907f);
	let m2: f32m3x3 = (
		f32m3x3(
			1.0f, 2.92338f, 59.4188f,
			20.3225f, -27.0302f, 222.592f,
			121.563f, 626.13f, 316.627f
		)
	);
	let m3: float2x2 = float2x2(0.0365463f, 3.32707f, 9.0632f, -9.04756f);
	let m4: f32m3x3 = (
		f32m3x3(
			1.0f, 3.59685f, -1.36772f,
			9.04401f, -16.3174f, 9.22949f,
			5.56589f, 19.7886f, -20.2123f
		)
	);
	var bias: f32 = (
		dot(mul(m1, x.xy), y.xy) / dot(mul(m2, x.xyw), y.xyw)
	);
	let scale: f32 = (
		dot(mul(m3, x.xy), y.xy) / dot(mul(m4, x.xzw), y.xyw)
	);
	// the reference's hack for zero specular reflectance
	bias *= saturate(specularColor.g * 50.0f);
	return specularColor * max(0.0f, scale) + max(0.0f, bias);
}
