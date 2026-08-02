module "util-material-openpbr-shared";
__exported import "util-types";

public struct OpenPbrMaterial {
	public f32 baseWeight;
	public f32v3 baseColor;
	public f32 baseMetalness;
	public f32 baseDiffuseRoughness;
	public f32 specularWeight;
	public f32v3 specularColor;
	public f32 specularRoughness;
	public f32 specularRoughnessAnisotropy;
	// not part of openpbr (which rotates via the geometry tangent instead);
	// gltf khr_materials_anisotropy rotation, radians counter-clockwise
	// from the tangent
	public f32 specularRoughnessAnisotropyRotation;
	public f32 specularIor;
	public f32 transmissionWeight;
	public f32v3 transmissionColor;
	public f32 transmissionDepth;
	public f32v3 transmissionScatter;
	public f32 transmissionScatterAnisotropy;
	public f32 transmissionDispersionScale;
	public f32 transmissionDispersionAbbeNumber;
	public f32 subsurfaceWeight;
	public f32v3 subsurfaceColor;
	public f32 subsurfaceRadius;
	public f32v3 subsurfaceRadiusScale;
	public f32 subsurfaceScatterAnisotropy;
	public f32 coatWeight;
	public f32v3 coatColor;
	public f32 coatRoughness;
	public f32 coatRoughnessAnisotropy;
	public f32 coatIor;
	public f32 coatDarkening;
	public f32 fuzzWeight;
	public f32v3 fuzzColor;
	public f32 fuzzRoughness;
	public f32 emissionLuminance;
	public f32v3 emissionColor;
	public f32 thinFilmWeight;
	public f32 thinFilmThickness;
	public f32 thinFilmIor;
	public f32 geometryOpacity;
	public f32 geometryThinWalled;

	public f32v3 geometryNormalTexture;
	public f32v4 geometryTangentTexture;
	public f32v3 geometryCoatNormalTexture;
	public f32v4 geometryCoatTangentTexture;

	// glTF extras...
	public f32 alphaCutoff;
	// (TODO REVIEW)
	// MOR_ALPHA_MODE_*
	public u32 alphaMode;
	// (TODO REVIEW)
};

