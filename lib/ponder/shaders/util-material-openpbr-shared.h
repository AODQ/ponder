#ifndef UTIL_MATERIAL_OPENPBR_SHARED_H
#define UTIL_MATERIAL_OPENPBR_SHARED_H

// -----------------------------------------------------------------------------
// -- OpenPbrMaterial
// -----------------------------------------------------------------------------

struct OpenPbrMaterial {
	float baseWeight;
	vec3 baseColor;
	float baseMetalness;
	float baseDiffuseRoughness;
	float specularWeight;
	vec3 specularColor;
	float specularRoughness;
	float specularRoughnessAnisotropy;
	// not part of openpbr (which rotates via the geometry tangent instead);
	// gltf khr_materials_anisotropy rotation, radians counter-clockwise
	// from the tangent
	float specularRoughnessAnisotropyRotation;
	float specularIor;
	float transmissionWeight;
	vec3 transmissionColor;
	float transmissionDepth;
	vec3 transmissionScatter;
	float transmissionScatterAnisotropy;
	float transmissionDispersionScale;
	float transmissionDispersionAbbeNumber;
	float subsurfaceWeight;
	vec3 subsurfaceColor;
	float subsurfaceRadius;
	vec3 subsurfaceRadiusScale;
	float subsurfaceScatterAnisotropy;
	float coatWeight;
	vec3 coatColor;
	float coatRoughness;
	float coatRoughnessAnisotropy;
	float coatIor;
	float coatDarkening;
	float fuzzWeight;
	vec3 fuzzColor;
	float fuzzRoughness;
	float emissionLuminance;
	vec3 emissionColor;
	float thinFilmWeight;
	float thinFilmThickness;
	float thinFilmIor;
	float geometryOpacity;
	float geometryThinWalled;

	vec3 geometryNormalTexture;
	vec4 geometryTangentTexture;
	vec3 geometryCoatNormalTexture;
	vec4 geometryCoatTangentTexture;

	// glTF extras...
	float alphaCutoff;
	// (TODO REVIEW)
	// MOR_ALPHA_MODE_*
	uint alphaMode;
	// (TODO REVIEW)
};

#endif
