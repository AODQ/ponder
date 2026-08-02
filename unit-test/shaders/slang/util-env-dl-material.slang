module "util-env-dl-material";
__exported import "util-types";
__exported import "util-material-openpbr-shared";

// shared material builder for the env-map direct-lighting estimator tests.
// env_direct_lighting.comp and env_direct_lighting_reference.comp must shade
// the *same* material or the comparison means nothing, so the field-by-field
// initialization lives here rather than being copied into both. every member
// is written explicitly: unwritten members are uninitialized gpu memory, not
// zero, and openPbrLobeSelection reads most of them.

public func envDlMaterial(
	baseColor: f32v3,
	baseMetalness: f32,
	specularRoughness: f32,
	specularIor: f32,
	coatWeight: f32,
	coatRoughness: f32,
	coatIor: f32,
) -> OpenPbrMaterial {
	OpenPbrMaterial mat;
	mat.baseWeight = 1.0f;
	mat.baseColor = baseColor;
	mat.baseMetalness = baseMetalness;
	mat.baseDiffuseRoughness = 0.0f;
	mat.specularWeight = 1.0f;
	mat.specularColor = f32v3(1.0f);
	mat.specularRoughness = specularRoughness;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularIor = specularIor;
	mat.transmissionWeight = 0.0f;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionScatter = f32v3(0.0f);
	mat.transmissionScatterAnisotropy = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 20.0f;
	mat.subsurfaceWeight = 0.0f;
	mat.subsurfaceColor = f32v3(1.0f);
	mat.subsurfaceRadius = 1.0f;
	mat.subsurfaceRadiusScale = f32v3(1.0f);
	mat.subsurfaceScatterAnisotropy = 0.0f;
	mat.coatWeight = coatWeight;
	mat.coatColor = f32v3(1.0f);
	mat.coatRoughness = coatRoughness;
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = coatIor;
	mat.coatDarkening = 1.0f;
	mat.fuzzWeight = 0.0f;
	mat.fuzzColor = f32v3(1.0f);
	mat.fuzzRoughness = 0.5f;
	mat.emissionLuminance = 0.0f;
	mat.emissionColor = f32v3(0.0f);
	mat.thinFilmWeight = 0.0f;
	mat.thinFilmThickness = 0.0f;
	mat.thinFilmIor = 1.5f;
	mat.geometryOpacity = 1.0f;
	mat.geometryThinWalled = 0.0f;
	mat.geometryNormalTexture = f32v3(0.0f, 0.0f, 1.0f);
	mat.geometryTangentTexture = f32v4(1.0f, 0.0f, 0.0f, 1.0f);
	mat.geometryCoatNormalTexture = f32v3(0.0f, 0.0f, 1.0f);
	mat.geometryCoatTangentTexture = f32v4(1.0f, 0.0f, 0.0f, 1.0f);
	mat.alphaCutoff = 0.5f;
	mat.alphaMode = 0u;
	return mat;
}

