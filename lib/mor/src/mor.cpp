#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <stb_image.h>


#include <meshoptimizer.h>

#include <mor/mor.hpp>
#include <srat/core-math.hpp>
#include <vkof/vkof.hpp>
#include <mor/mor-shared.h>

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

#include <strings.h> // strcasecmp

// ----------------------------------------------------------------------------
// -- private types
// ----------------------------------------------------------------------------

namespace {

struct SamplerKey
{
	vkof::SamplerFilter magFilter;
	vkof::SamplerFilter minFilter;
	vkof::SamplerAddressMode addressU;
	vkof::SamplerAddressMode addressV;
	vkof::SamplerMipmapMode mipmapMode;
	f32 maxAnisotropy;
	bool operator==(SamplerKey const & o) const {
		return (
			magFilter == o.magFilter
			&& minFilter == o.minFilter
			&& addressU == o.addressU
			&& addressV == o.addressV
			&& mipmapMode == o.mipmapMode
			&& maxAnisotropy == o.maxAnisotropy
		);
	}
};

struct SamplerKeyHash
{
	size_t operator()(SamplerKey const & k) const {
		size_t h = 0;
		auto mix = [&](size_t v) { h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2); };
		mix((size_t)k.magFilter);
		mix((size_t)k.minFilter);
		mix((size_t)k.addressU);
		mix((size_t)k.addressV);
		mix((size_t)k.mipmapMode);
		mix((size_t)(k.maxAnisotropy * 4.0f));
		return h;
	}
};

static std::unordered_map<SamplerKey, vkof::Sampler, SamplerKeyHash> sSamplerCache;
static vkof::Sampler sImguiDisplaySampler = { 0u };

// the same gltf texture can be requested as srgb (color) and unorm (data),
// which are distinct images; the cache key must carry both
struct TextureKey
{
	cgltf_texture const * texture;
	bool srgb;
	bool operator==(TextureKey const & o) const {
		return texture == o.texture && srgb == o.srgb;
	}
};

struct TextureKeyHash
{
	size_t operator()(TextureKey const & k) const {
		return (
			std::hash<void const *>()(k.texture)
			^ (k.srgb ? (size_t)0x9e3779b9u : (size_t)0u)
		);
	}
};

// a texture's return value from load_texture: the bindless image/sampler
// handle plus an index into the scene's GpuMorUvTransform table
struct TextureRef
{
	u32 handle;
	u32 uvTransform;
};

// dedup key for gltf KHR_texture_transform; raw offset/rotation/scale, not
// the precomputed matrix, since two authored transforms are only the same
// transform if these source fields match exactly
struct UvTransformKey
{
	f32 offsetX, offsetY, rotation, scaleX, scaleY;
	bool operator==(UvTransformKey const & o) const {
		return (
			offsetX == o.offsetX
			&& offsetY == o.offsetY
			&& rotation == o.rotation
			&& scaleX == o.scaleX
			&& scaleY == o.scaleY
		);
	}
};

struct UvTransformKeyHash
{
	size_t operator()(UvTransformKey const & k) const {
		size_t h = 0;
		auto mix = [&](f32 v) {
			h ^= std::hash<f32>()(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
		};
		mix(k.offsetX);
		mix(k.offsetY);
		mix(k.rotation);
		mix(k.scaleX);
		mix(k.scaleY);
		return h;
	}
};

struct ImplScene {
	std::vector<f32v3> positions;
	std::vector<GpuMorVertexAttribute> attributes;
	std::vector<u32> meshletVerts;
	std::vector<u8> meshletTris;
	std::vector<GpuMorMeshlet> meshlets;
	std::vector<GpuMorInstance> instances;

	std::string gltfDir;
	// only valid during scene_load; needed to resolve extension image
	// indices (MSFT_texture_dds) that cgltf keeps as raw json
	cgltf_data const * data { nullptr };
	std::unordered_map<TextureKey, u32, TextureKeyHash> textureHandles;
	std::unordered_map<TextureKey, vkof::Image, TextureKeyHash> textureImages;
	std::unordered_map<TextureKey, SamplerKey, TextureKeyHash> textureSamplerKeys;
	std::vector<vkof::Image> images;
	std::vector<std::string> imageNames;
	std::vector<ImTextureID> imguiIds;

	std::unordered_map<cgltf_material const *, u32> materialIndices;

	std::vector<GpuMorMaterial> materials;
	std::vector<std::string> materialNames;

	// index 0 is the reserved identity transform, seeded in load_texture on
	// first use
	std::vector<GpuMorUvTransform> uvTransforms;
	std::unordered_map<UvTransformKey, u32, UvTransformKeyHash> uvTransformCache;

	// true once any material in the scene is loaded with a gltf blend mode
	// other than opaque (mask or blend); drives whether the scene's blas
	// can be marked opaque
	bool hasNonOpaqueMaterial { false };

	// old handle -> new handle from the last scene_set_anisotropy call
	std::unordered_map<u32, u32> lastHandleRemap;
};

struct ImplGpuMaterials
{
	vkof::Buffer buffer;
	std::vector<GpuMorMaterial> cpu;
};

struct ImplGpuScene {
	vkof::Buffer positions;
	vkof::Buffer attributes;
	vkof::Buffer meshletVerts;
	vkof::Buffer meshletTris;
	vkof::Buffer meshlets;
	vkof::Buffer instances;
	vkof::Buffer materials;
	vkof::Buffer uvTransforms;
	vkof::Buffer textures;
	vkof::Buffer flatIndices;
	vkof::Buffer flatMeshlets;
	u32 meshletCount;
	u32 vertexCount;
	u32 triangleCount;
};

static constexpr u32 skMaxMeshletVerts = 64;
static constexpr u32 skMaxMeshletTris = 124;

} // namespace

// ----------------------------------------------------------------------------
// -- private helpers
// ----------------------------------------------------------------------------

// resolves the KHR_texture_transform on this texture_view (if any) to an
// index into s.uvTransforms, deduped by the raw offset/rotation/scale.
// this is per texture_view (usage site), not per cgltf_texture, because the
// same underlying texture can be reused by multiple materials with
// different transforms (e.g. the same tileable normal map reused at
// different tiling densities)
static u32 load_uv_transform(
	ImplScene & s,
	cgltf_texture_view const & texture
) {
	if (!texture.has_transform) { return 0u; }
	// only uv set 0 is ever loaded (see load_primitive), so a texcoord
	// override in the transform has nothing to switch to and is ignored
	cgltf_texture_transform const & xf = texture.transform;
	UvTransformKey const key {
		.offsetX = xf.offset[0],
		.offsetY = xf.offset[1],
		.rotation = xf.rotation,
		.scaleX = xf.scale[0],
		.scaleY = xf.scale[1],
	};
	auto const cacheIt = s.uvTransformCache.find(key);
	if (cacheIt != s.uvTransformCache.end()) { return cacheIt->second; }

	if (s.uvTransforms.empty()) {
		// index 0 is the reserved identity transform
		s.uvTransforms.push_back({
			.basisU = { 1.0f, 0.0f },
			.basisV = { 0.0f, 1.0f },
			.offset = { 0.0f, 0.0f },
		});
	}
	f32 const c = std::cos(xf.rotation);
	f32 const sn = std::sin(xf.rotation);
	u32 const index = (u32)s.uvTransforms.size();
	s.uvTransforms.push_back({
		.basisU = { c * xf.scale[0], sn * xf.scale[0] },
		.basisV = { -sn * xf.scale[1], c * xf.scale[1] },
		.offset = { xf.offset[0], xf.offset[1] },
	});
	s.uvTransformCache.emplace(key, index);
	return index;
}

// a bc7 dds file cracked open: dimensions, baked mip count and a view of
// the tightly packed block payload
struct DdsData
{
	u32 width;
	u32 height;
	u32 mipLevels;
	srat::slice<u8 const> payload;
};

// parses a dx10-header dds file; only bc7 (dxgi 98/99) is accepted since
// that is the one block format the renderer uploads natively. srgb-ness is
// decided by the requesting material slot, not the file, so both dxgi
// variants land in the same payload view
static bool dds_parse(std::vector<u8> const & file, DdsData & out)
{
	auto const rd32 = [&](size_t const off) {
		u32 v;
		std::memcpy(&v, file.data() + off, 4);
		return v;
	};
	constexpr size_t skHeaderSize = 148; // magic + DDS_HEADER + DDS_HEADER_DXT10
	if (file.size() < skHeaderSize) { return false; }
	if (std::memcmp(file.data(), "DDS ", 4) != 0) { return false; }
	if (rd32(4) != 124) { return false; }
	u32 const height = rd32(12);
	u32 const width = rd32(16);
	u32 const mipLevels = std::max(rd32(28), 1u);
	if (std::memcmp(file.data() + 84, "DX10", 4) != 0) { return false; }
	u32 const dxgiFormat = rd32(128);
	if (dxgiFormat != 98 && dxgiFormat != 99) { return false; }

	u64 payloadSize = 0;
	for (u32 mip = 0; mip < mipLevels; ++mip) {
		u32 const mipW = std::max(width >> mip, 1u);
		u32 const mipH = std::max(height >> mip, 1u);
		payloadSize += (u64)((mipW + 3u) / 4u) * ((mipH + 3u) / 4u) * 16u;
	}
	if (file.size() < skHeaderSize + payloadSize) { return false; }

	out = DdsData {
		.width = width,
		.height = height,
		.mipLevels = mipLevels,
		.payload = srat::slice<u8 const>(
			file.data() + skHeaderSize, payloadSize
		),
	};
	return true;
}

// cgltf does not parse MSFT_texture_dds, so the dds image index has to be
// fished out of the retained raw extension json ({"source": N})
static cgltf_image const * texture_dds_image(
	ImplScene const & s,
	cgltf_texture const * tex
) {
	if (!s.data) { return nullptr; }
	for (cgltf_size i = 0; i < tex->extensions_count; ++i) {
		cgltf_extension const & ext = tex->extensions[i];
		if (!ext.name || std::strcmp(ext.name, "MSFT_texture_dds") != 0) {
			continue;
		}
		if (!ext.data) { return nullptr; }
		char const * src = std::strstr(ext.data, "\"source\"");
		if (!src) { return nullptr; }
		src = std::strchr(src, ':');
		if (!src) { return nullptr; }
		long const index = std::strtol(src + 1, nullptr, 10);
		if (index < 0 || (cgltf_size)index >= s.data->images_count) {
			return nullptr;
		}
		return &s.data->images[index];
	}
	return nullptr;
}

static bool uri_is_dds(char const * uri) {
	if (!uri) { return false; }
	size_t const len = std::strlen(uri);
	return len >= 4 && strcasecmp(uri + len - 4, ".dds") == 0;
}

static TextureRef load_texture(
	ImplScene & s,
	cgltf_texture_view const & texture,
	bool const srgb
) {
	if (!texture.texture || !texture.texture->image) { return { 0u, 0u }; }
	auto const & tex = texture.texture;

	u32 const uvTransform = load_uv_transform(s, texture);

	TextureKey const texKey { .texture = tex, .srgb = srgb };
	auto const it = s.textureHandles.find(texKey);
	if (it != s.textureHandles.end()) { return { it->second, uvTransform }; }

	cgltf_image const * img = tex->image;

	// -- dds path: an MSFT_texture_dds source, or a base image that is
	// itself a dds; bc7 blocks upload as-is with their baked mip chain
	vkof::Image image { 0u };
	bool loaded = false;
	{
		cgltf_image const * ddsImg = texture_dds_image(s, tex);
		if (!ddsImg && uri_is_dds(img->uri)) { ddsImg = img; }
		if (ddsImg && ddsImg->uri && strncmp(ddsImg->uri, "data:", 5) != 0) {
			std::string const fullPath = (
				std::filesystem::path(s.gltfDir) / ddsImg->uri
			).string();
			std::ifstream file(fullPath, std::ios::binary);
			if (file) {
				std::vector<u8> const bytes(
					(std::istreambuf_iterator<char>(file)),
					std::istreambuf_iterator<char>()
				);
				DdsData dds;
				if (dds_parse(bytes, dds)) {
					image = vkof::image_create({
						.width = dds.width,
						.height = dds.height,
						.format = (
							srgb
							? vkof::ImageFormat::bc7_srgb
							: vkof::ImageFormat::bc7_unorm
						),
						.mipLevels = dds.mipLevels,
						.optInitialData = dds.payload,
					});
					loaded = true;
					img = ddsImg;
				} else {
					printf("mor: unsupported dds file '%s'\n", fullPath.c_str());
				}
			}
		}
	}

	if (!loaded) {
		int w, h, channels;
		stbi_uc * pixels = nullptr;

		if (img->buffer_view) {
			u8 const * encoded = (
				(u8 const *)img->buffer_view->buffer->data
				+ img->buffer_view->offset
			);
			int const encodedLen = (int)img->buffer_view->size;
			pixels = stbi_load_from_memory(encoded, encodedLen, &w, &h, &channels, 4);
		} else if (img->uri && strncmp(img->uri, "data:", 5) != 0) {
			std::string const fullPath = (std::filesystem::path(s.gltfDir) / img->uri).string();
			pixels = stbi_load(fullPath.c_str(), &w, &h, &channels, 4);
		}

		if (!pixels) {
			printf(
				"mor: failed to load texture '%s'\n",
				img->uri ? img->uri : "[embedded]"
			);
			return { 0u, uvTransform };
		}

		u32 const mipLevels = (
			(u32)std::floor(std::log2((f32)std::max(w, h))) + 1u
		);
		image = vkof::image_create({
			.width = (u32)w,
			.height = (u32)h,
			.format = (
				srgb
				? vkof::ImageFormat::r8g8b8a8_srgb
				: vkof::ImageFormat::r8g8b8a8_unorm
			),
			.mipLevels = mipLevels,
			.optInitialData = srat::slice<u8 const>(
				pixels, (u64)w * h * 4
			),
		});
		stbi_image_free(pixels);
		vkof::image_generate_mipmaps(image);
	}

	auto const toFilter = [](int gl) -> vkof::SamplerFilter {
		if (gl == 9728 || gl == 9984 || gl == 9986) {
			return vkof::SamplerFilter::nearest;
		}
		return vkof::SamplerFilter::linear;
	};
	auto const toWrap = [](int gl) -> vkof::SamplerAddressMode {
		if (gl == 33071) return vkof::SamplerAddressMode::clamp_to_edge;
		if (gl == 33648) return vkof::SamplerAddressMode::mirrored_repeat;
		return vkof::SamplerAddressMode::repeat;
	};

	cgltf_sampler const * cgSamp = tex->sampler;
	SamplerKey const key {
		.magFilter = (
			cgSamp
			? toFilter(cgSamp->mag_filter)
			: vkof::SamplerFilter::linear
		),
		.minFilter = (
			cgSamp
			? toFilter(cgSamp->min_filter)
			: vkof::SamplerFilter::linear
		),
		.addressU = (
			cgSamp
			? toWrap(cgSamp->wrap_s)
			: vkof::SamplerAddressMode::repeat
		),
		.addressV = (
			cgSamp
			? toWrap(cgSamp->wrap_t)
			: vkof::SamplerAddressMode::repeat
		),
		.mipmapMode = vkof::SamplerMipmapMode::linear,
		.maxAnisotropy = 16.0f,
	};
	auto cacheIt = sSamplerCache.find(key);
	if (cacheIt == sSamplerCache.end()) {
		vkof::Sampler const s = vkof::sampler_create({
			.magFilter = key.magFilter,
			.minFilter = key.minFilter,
			.addressModeU = key.addressU,
			.addressModeV = key.addressV,
			.addressModeW = vkof::SamplerAddressMode::repeat,
			.mipmapMode = key.mipmapMode,
			.maxAnisotropy = key.maxAnisotropy,
		});
		cacheIt = sSamplerCache.emplace(key, s).first;
	}
	vkof::Sampler const sampler = cacheIt->second;

	u32 const handle = vkof::image_sampler_handle({
		.image = image,
		.sampler = sampler,
	});

	s.images.push_back(image);
	{
		std::string name;
		if (tex->name && tex->name[0] != '\0') {
			name = tex->name;
		} else if (img->name && img->name[0] != '\0') {
			name = img->name;
		} else if (img->uri) {
			name = std::filesystem::path(img->uri).filename().string();
		} else {
			name = "[texture " + std::to_string(s.imageNames.size()) + "]";
		}
		s.imageNames.push_back(std::move(name));
	}
	s.textureHandles.emplace(texKey, handle);
	s.textureImages.emplace(texKey, image);
	s.textureSamplerKeys.emplace(texKey, key);
	return { handle, uvTransform };
}

// gltf permits a roughness factor of exactly 0.0 (a perfect mirror), but the
// ggx distribution/visibility terms have a removable singularity there that
// this renderer does not special-case; floor every roughness factor read
// from a gltf asset just above zero so it stays a well-defined (if very
// sharp) microfacet lobe instead of a NaN-producing one
static constexpr f32 skMinRoughness = 1e-3f;

static f32 material_clamp_roughness(f32 const roughness) {
	return std::max(roughness, skMinRoughness);
}

static GpuMorMaterial material_load_default()
{
	// defaults set by OpenPBR
	auto const rgb1 = f32v3 { 1.0f, 1.0f, 1.0f };
	auto const rgb0 = f32v3 { 0.0f, 0.0f, 0.0f };
	GpuMorMaterial material;
	memset(&material, 0, sizeof(material));
	material.baseWeight.r = 1.0f;
	material.baseColor.rgb = f32v3 { 0.8f, 0.8f, 0.8f, };
	material.baseMetalness.r = 0.0f;
	material.baseDiffuseRoughness.r = 0.0f;
	material.specularWeight.r = 1.0f;
	material.specularColor.rgb = rgb1;
	material.specularRoughness.r = 0.3f;
	material.specularRoughnessAnisotropy.r = 0.0f;
	material.specularRoughnessAnisotropyRotation.r = 0.0f;
	material.specularIor.r = 1.5f;
	material.transmissionWeight.r = 0.0f;
	material.transmissionColor.rgb = rgb1;
	material.transmissionDepth.r = 0.0f;
	material.transmissionScatter.rgb = rgb0;
	material.transmissionScatterAnisotropy.r = 0.0f;
	material.transmissionDispersionScale.r = 0.0f;
	material.transmissionDispersionAbbeNumber.r = 20.0f;
	material.subsurfaceWeight.r = 0.0f;
	material.subsurfaceColor.rgb = f32v3 { 0.8f, 0.8f, 0.8f };
	material.subsurfaceRadius.r = 1.0f;
	material.subsurfaceRadiusScale.rgb = f32v3 { 1.0f, 0.5f, 0.25f };
	material.subsurfaceScatterAnisotropy.r = 0.0f;
	material.coatWeight.r = 0.0f;
	material.coatColor.rgb = rgb1;
	material.coatRoughness.r = 0.0f;
	material.coatRoughnessAnisotropy.r = 0.0f;
	material.coatIor.r = 1.6f;
	material.coatDarkening.r = 1.0f;
	material.fuzzWeight.r = 0.0f;
	material.fuzzColor.rgb = rgb1;
	material.fuzzRoughness.r = 0.5f;
	material.emissionLuminance.r = 0.0f;
	material.emissionColor.rgb = rgb1;
	material.thinFilmWeight.r = 0.0f;
	material.thinFilmThickness.r = 0.5f;
	material.thinFilmIor.r = 1.4f;
	material.geometryOpacity.r = 1.0f;
	material.geometryThinWalled.r = 0.0f;
	return material;
}

static void material_load_pbr_metallic_roughness(
	ImplScene & s,
	cgltf_pbr_metallic_roughness  const & mr,
	GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_texture_view base_color_texture;
	// cgltf_texture_view metallic_roughness_texture;
	// cgltf_float base_color_factor[4];
	// cgltf_float metallic_factor;
	// cgltf_float roughness_factor;
	// OpenPBR:
	// base-weight, base-color, base-metalness, base-diffuse-roughness,
	// geometry-opacity

	// -- base color
	if (mr.base_color_texture.texture) {
		TextureRef const t = load_texture(s, mr.base_color_texture, true);
		out.baseColor.texture = t.handle;
		out.baseColor.uvTransform = t.uvTransform;
		// alpha is linear even in srgb images; reuse the srgb texture
		out.geometryOpacity.texture = t.handle;
		out.geometryOpacity.uvTransform = t.uvTransform;
		// opacity is in A
		out.geometryOpacity.swizzle = 3;
	}

	// -- metallic
	if (mr.metallic_roughness_texture.texture) {
		TextureRef const t = load_texture(s, mr.metallic_roughness_texture, false);
		out.baseMetalness.texture = t.handle;
		out.baseMetalness.uvTransform = t.uvTransform;
		out.baseMetalness.swizzle = 2; // [AO, Roughness, Metalness] in glTF
	}

	// -- roughness
	if (mr.metallic_roughness_texture.texture) {
		TextureRef const t = load_texture(s, mr.metallic_roughness_texture, false);
		out.specularRoughness.texture = t.handle;
		out.specularRoughness.uvTransform = t.uvTransform;
		out.specularRoughness.swizzle = 1; // [AO, Roughness, Metalness] in glTF
	}

	out.baseMetalness.r = mr.metallic_factor;
	out.specularRoughness.r = material_clamp_roughness(mr.roughness_factor);
	out.baseColor.rgb = f32v3 {
		mr.base_color_factor[0],
		mr.base_color_factor[1],
		mr.base_color_factor[2],
	};
	out.geometryOpacity.r = mr.base_color_factor[3];
}

static void material_load_pbr_specular_glossiness(
	ImplScene & s,
	cgltf_pbr_specular_glossiness const & sg,
	GpuMorMaterial & out
) {
	// GLTF (KHR_materials_pbrSpecularGlossiness, archived):
	// cgltf_texture_view diffuse_texture;
	// cgltf_texture_view specular_glossiness_texture;
	// cgltf_float diffuse_factor[4];
	// cgltf_float specular_factor[3];
	// cgltf_float glossiness_factor;
	// OpenPBR:
	// base-color, geometry-opacity, specular-color, specular-roughness
	//
	// approximate mapping: diffuse drives the dielectric base, specular rgb
	// tints the dielectric fresnel, and roughness = 1 - glossiness via the
	// invert swizzle bit. spec-gloss metals (colored specular over a black
	// diffuse) are not reconstructed into a metallic lobe

	// -- diffuse
	if (sg.diffuse_texture.texture) {
		TextureRef const t = load_texture(s, sg.diffuse_texture, true);
		out.baseColor.texture = t.handle;
		out.baseColor.uvTransform = t.uvTransform;
		// alpha is linear even in srgb images; reuse the srgb texture
		out.geometryOpacity.texture = t.handle;
		out.geometryOpacity.uvTransform = t.uvTransform;
		// opacity is in A
		out.geometryOpacity.swizzle = 3;
	}

	// -- specular + glossiness
	if (sg.specular_glossiness_texture.texture) {
		TextureRef const t = load_texture(s, sg.specular_glossiness_texture, true);
		out.specularColor.texture = t.handle;
		out.specularColor.uvTransform = t.uvTransform;
		// glossiness is in A (linear even in srgb images)
		out.specularRoughness.texture = t.handle;
		out.specularRoughness.uvTransform = t.uvTransform;
		out.specularRoughness.swizzle = 3;
	}
	// resolved value is 1 - glossiness_factor * gloss_texture.a, textured
	// or not; the shader floors it to its minimum roughness
	out.specularRoughness.swizzle |= MOR_MATERIAL_SWIZZLE_INVERT;
	out.specularRoughness.r = sg.glossiness_factor;

	out.baseColor.rgb = f32v3 {
		sg.diffuse_factor[0],
		sg.diffuse_factor[1],
		sg.diffuse_factor[2],
	};
	out.geometryOpacity.r = sg.diffuse_factor[3];
	out.specularColor.rgb = f32v3 {
		sg.specular_factor[0],
		sg.specular_factor[1],
		sg.specular_factor[2],
	};
}

static void material_load_clearcoat(
	ImplScene & s,
	cgltf_clearcoat const & cc,
	GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_texture_view clearcoat_texture;
	// cgltf_texture_view clearcoat_roughness_texture;
	// cgltf_texture_view clearcoat_normal_texture;
	// cgltf_float clearcoat_factor;
	// cgltf_float clearcoat_roughness_factor;
	// OpenPBR:
	// coat-weight, coat-color, coat-roughness,
	// coat-roughness-anisotropy, coat-ior, coat-darkening
	// u32 geometryCoatNormalTexture;
	// u32 geometryCoatTangentTexture;

	if (cc.clearcoat_texture.texture) {
		TextureRef const t = load_texture(s, cc.clearcoat_texture, false);
		out.coatWeight.texture = t.handle;
		out.coatWeight.uvTransform = t.uvTransform;
		// clearcoat strength is in R
		out.coatWeight.swizzle = 0;
	}
	if (cc.clearcoat_roughness_texture.texture) {
		TextureRef const t = load_texture(s, cc.clearcoat_roughness_texture, false);
		out.coatRoughness.texture = t.handle;
		out.coatRoughness.uvTransform = t.uvTransform;
		// clearcoat roughness is in G
		out.coatRoughness.swizzle = 1;
	}
	if (cc.clearcoat_normal_texture.texture) {
		TextureRef const t = load_texture(s, cc.clearcoat_normal_texture, false);
		out.geometryCoatNormalTexture = t.handle;
		out.geometryCoatNormalUvTransform = t.uvTransform;
	}
	out.coatWeight.r = cc.clearcoat_factor;
	out.coatRoughness.r = material_clamp_roughness(cc.clearcoat_roughness_factor);
}

static void material_load_diffuse_transmission(
	ImplScene & s,
	cgltf_diffuse_transmission const & dt,
	GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_texture_view diffuse_transmission_texture;
	// cgltf_float diffuse_transmission_factor;
	// cgltf_float diffuse_transmission_color_factor[3];
	// cgltf_texture_view diffuse_transmission_color_texture;
	// OpenPBR:
	// subsurfaceWeight;
	// subsurfaceColor;
	// subsurfaceScatterAnisotropy;
	// geometryThinWalled;

	// openpbr's thin-walled subsurface sheet splits S = subsurface_color
	// into 1/2 S (1 - g) reflected and 1/2 S (1 + g) transmitted, and the
	// subsurface weight blends the sheet against the ordinary base diffuse
	// lobe. g = 1 sends the whole sheet to the transmitted side, which
	// reproduces the gltf split exactly:
	//   reflected   = (1 - factor) * baseColor   <- base diffuse lobe
	//   transmitted = factor * colorFactor       <- sheet, all transmitting
	out.subsurfaceScatterAnisotropy.r = 1.0f;
	if (dt.diffuse_transmission_texture.texture) {
		TextureRef const t = (
			load_texture(s, dt.diffuse_transmission_texture, false)
		);
		out.subsurfaceWeight.texture = t.handle;
		out.subsurfaceWeight.uvTransform = t.uvTransform;
		// diffuse transmission strength is in A
		out.subsurfaceWeight.swizzle = 3;
	}
	if (dt.diffuse_transmission_color_texture.texture) {
		TextureRef const t = (
			load_texture(s, dt.diffuse_transmission_color_texture, true)
		);
		out.subsurfaceColor.texture = t.handle;
		out.subsurfaceColor.uvTransform = t.uvTransform;
	}
	out.subsurfaceWeight.r = dt.diffuse_transmission_factor;
	out.subsurfaceColor.rgb = f32v3 {
		dt.diffuse_transmission_color_factor[0],
		dt.diffuse_transmission_color_factor[1],
		dt.diffuse_transmission_color_factor[2],
	};
}

static void material_load_transmission(
	ImplScene & s,
	cgltf_transmission const & tr,
	GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_texture_view transmission_texture;
	// cgltf_float transmission_factor;
	// OpenPBR:
	// transmissionWeight;
	// transmissionColor;
	// transmissionDepth;
	// transmissionScatter;
	// transmissionScatterAnisotropy;
	// transmissionDispersionScale;
	// transmissionDispersionAbbeNumber;

	if (tr.transmission_texture.texture) {
		TextureRef const t = load_texture(s, tr.transmission_texture, false);
		out.transmissionWeight.texture = t.handle;
		out.transmissionWeight.uvTransform = t.uvTransform;
		// transmission strength is in R
		out.transmissionWeight.swizzle = 0;
	}
	out.transmissionWeight.r = tr.transmission_factor;
	// gltf tints transmitted light by base color (spec: specular_btdf *
	// base_color); openpbr keeps a separate transmission_color, so mirror
	// it, texture included. a volume overwrites this with attenuationColor
	out.transmissionColor = out.baseColor;
}

static void material_load_volume(
	[[maybe_unused]] ImplScene & s,
	cgltf_volume const & vol,
	GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_texture_view thickness_texture;
	// cgltf_float thickness_factor;
	// cgltf_float attenuation_color[3];
	// cgltf_float attenuation_distance;
	// OpenPBR:
	// transmissionWeight;
	// transmissionColor;
	// transmissionDepth;
	// transmissionScatter;
	// transmissionScatterAnisotropy;
	// transmissionDispersionScale;
	// transmissionDispersionAbbeNumber;

	out.transmissionColor.rgb = f32v3 {
		vol.attenuation_color[0],
		vol.attenuation_color[1],
		vol.attenuation_color[2],
	};
	// attenuation is a plain factor; drop any base color texture the
	// transmission load mirrored in
	out.transmissionColor.texture = 0u;
	out.transmissionColor.uvTransform = 0u;
	out.transmissionDepth.r = vol.attenuation_distance;
}

// KHR_materials_volume_scatter postdates cgltf v1.15, so it is read out of
// the raw extension json. it carries its own attenuation, which wins over
// KHR_materials_volume's
struct VolumeScatter {
	bool present;
	bool hasMultiscatterColor;
	bool hasAttenuationColor;
	bool hasAttenuationDistance;
	f32v3 multiscatterColor;
	f32v3 attenuationColor;
	f32 attenuationDistance;
};

static bool json_read_floats(
	char const * const data, char const * const key, f32 * const out,
	u32 const count
) {
	char const * const found = strstr(data, key);
	if (found == nullptr) {
		return false;
	}
	char const * cursor = found + strlen(key);
	// a scalar has no bracket; an array's values start after it
	char const * const bracket = strchr(cursor, '[');
	char const * const colon = strchr(cursor, ':');
	if (colon == nullptr) {
		return false;
	}
	cursor = (count > 1u && bracket != nullptr) ? bracket + 1 : colon + 1;
	for (u32 i = 0u; i < count; ++i) {
		char * end = nullptr;
		f32 const value = strtof(cursor, &end);
		if (end == cursor) {
			return false;
		}
		out[i] = value;
		cursor = end;
		while (*cursor == ',' || *cursor == ' ') {
			++cursor;
		}
	}
	return true;
}

static VolumeScatter material_volume_scatter(cgltf_material const & mat) {
	VolumeScatter scatter = {};
	for (cgltf_size i = 0u; i < mat.extensions_count; ++i) {
		cgltf_extension const & ext = mat.extensions[i];
		if (ext.name == nullptr || ext.data == nullptr) {
			continue;
		}
		if (strcmp(ext.name, "KHR_materials_volume_scatter") != 0) {
			continue;
		}
		scatter.present = true;
		f32 rgb[3] = { 1.0f, 1.0f, 1.0f };
		if (json_read_floats(ext.data, "\"multiscatterColorFactor\"", rgb, 3u)) {
			scatter.hasMultiscatterColor = true;
			scatter.multiscatterColor = f32v3 { rgb[0], rgb[1], rgb[2] };
		}
		if (json_read_floats(ext.data, "\"attenuationColor\"", rgb, 3u)) {
			scatter.hasAttenuationColor = true;
			scatter.attenuationColor = f32v3 { rgb[0], rgb[1], rgb[2] };
		}
		f32 distance = 0.0f;
		if (json_read_floats(ext.data, "\"attenuationDistance\"", &distance, 1u)) {
			scatter.hasAttenuationDistance = true;
			scatter.attenuationDistance = distance;
		}
		break;
	}
	return scatter;
}

// diffuse transmission into a real interior: the walk runs instead of the
// thin sheet. the tint has to land on subsurfaceColor, not on the radius --
// the walk reads color as the observed diffuse reflectance and derives
// absorption from it, so a white color absorbs nothing and renders
// colorless however chromatic the mean free path is
static void material_load_diffuse_transmission_volume(
	[[maybe_unused]] ImplScene & s,
	cgltf_material const & mat,
	GpuMorMaterial & out
) {
	cgltf_volume const & vol = mat.volume;
	VolumeScatter const scatter = material_volume_scatter(mat);
	// volume_scatter's multiscatter albedo is the real scattering color;
	// plain volume only has absorption, so fall back to that
	f32v3 tint = f32v3 {
		vol.attenuation_color[0],
		vol.attenuation_color[1],
		vol.attenuation_color[2],
	};
	if (scatter.hasAttenuationColor) {
		tint = scatter.attenuationColor;
	}
	if (scatter.hasMultiscatterColor) {
		tint = scatter.multiscatterColor;
	}
	out.subsurfaceColor.rgb = f32v3 {
		out.subsurfaceColor.rgb.x * tint.x,
		out.subsurfaceColor.rgb.y * tint.y,
		out.subsurfaceColor.rgb.z * tint.z,
	};
	// per-channel mean free path, spec's sigma_t = -log(c) / d inverted.
	// radius carries the absolute length, radiusScale the 0..1 modulation
	// the rest of the codebase assumes
	f32v3 const attenuation = (
		scatter.hasAttenuationColor
		? scatter.attenuationColor
		: f32v3 {
			vol.attenuation_color[0],
			vol.attenuation_color[1],
			vol.attenuation_color[2],
		}
	);
	// the sheet's all-transmit g=1 is meaningless to the walk, which reads
	// this as the henyey-greenstein asymmetry; gltf authors no such
	// parameter, so the phase function is isotropic
	out.subsurfaceScatterAnisotropy.r = 0.0f;
	f32 const distance = (
		scatter.hasAttenuationDistance
		? scatter.attenuationDistance
		: vol.attenuation_distance
	);
	// unauthored distance is +inf/FLT_MAX; no length to carry over. world
	// space per spec, so no node-scale correction
	bool const hasAttenuation = (
		std::isfinite(distance) && distance > 0.0f && distance < FLT_MAX
	);
	if (!hasAttenuation) {
		out.subsurfaceRadiusScale.rgb = f32v3 { 1.0f, 1.0f, 1.0f };
		return;
	}
	// attenuation 1 is an infinite mean free path; floor caps it at 1e3
	auto const meanFreePath = [&](f32 const channel) -> f32 {
		f32 const clamped = std::clamp(channel, 1e-4f, 1.0f);
		return distance / std::max(-std::log(clamped), 1e-3f);
	};
	f32v3 const mfp = f32v3 {
		meanFreePath(attenuation.x),
		meanFreePath(attenuation.y),
		meanFreePath(attenuation.z),
	};
	f32 const longest = std::max({ mfp.x, mfp.y, mfp.z });
	out.subsurfaceRadius.r = longest;
	out.subsurfaceRadiusScale.rgb = f32v3 {
		mfp.x / longest, mfp.y / longest, mfp.z / longest,
	};
}

static void material_load_ior(
	[[maybe_unused]] ImplScene & s,
	cgltf_material const & mat,
	GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_float ior;
	// OpenPBR:
	// specularIor;

	// the gltf ior applies to the base dielectric only; the gltf clearcoat
	// ior is fixed at 1.5 and the iridescence ior has its own extension field
	out.specularIor.r = mat.ior.ior;
}

static void material_load_emission(
	[[maybe_unused]] ImplScene & s,
	cgltf_emissive_strength const & em,
	GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_float emissive_strength;
	// OpenPBR:
	// emissionLuminance;
	// emissionColor;

	out.emissionLuminance.r = em.emissive_strength;
}

static void material_load_specular(
	ImplScene & s,
	cgltf_specular const & sp,
	GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_texture_view specular_texture;
	// cgltf_texture_view specular_color_texture;
	// cgltf_float specular_factor;
	// cgltf_float specular_color_factor[3];
	// OpenPBR:
	// specularWeight;
	// specularColor;
	// specularRoughness;
	// specularRoughnessAnisotropy;
	// specularIor;

	if (sp.specular_texture.texture) {
		TextureRef const t = load_texture(s, sp.specular_texture, false);
		out.specularWeight.texture = t.handle;
		out.specularWeight.uvTransform = t.uvTransform;
		// specular strength is in A
		out.specularWeight.swizzle = 3;
	}
	if (sp.specular_color_texture.texture) {
		TextureRef const t = load_texture(s, sp.specular_color_texture, true);
		out.specularColor.texture = t.handle;
		out.specularColor.uvTransform = t.uvTransform;
	}
	out.specularWeight.r = sp.specular_factor;
	out.specularColor.rgb = f32v3 {
		sp.specular_color_factor[0],
		sp.specular_color_factor[1],
		sp.specular_color_factor[2],
	};
}

static void material_load_sheen(
	[[maybe_unused]] ImplScene & s,
	[[maybe_unused]] cgltf_sheen const & sh,
	[[maybe_unused]] GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_texture_view sheen_color_texture;
	// cgltf_float sheen_color_factor[3];
	// cgltf_texture_view sheen_roughness_texture;
	// cgltf_float sheen_roughness_factor;
	// OpenPBR:
	// GpuMorMaterialComponent1 fuzzWeight;
	// GpuMorMaterialComponent3 fuzzColor;
	// GpuMorMaterialComponent1 fuzzRoughness;

	if (sh.sheen_color_texture.texture) {
		TextureRef const t = load_texture(s, sh.sheen_color_texture, true);
		out.fuzzColor.texture = t.handle;
		out.fuzzColor.uvTransform = t.uvTransform;
	}
	if (sh.sheen_roughness_texture.texture) {
		TextureRef const t = load_texture(s, sh.sheen_roughness_texture, false);
		out.fuzzRoughness.texture = t.handle;
		out.fuzzRoughness.uvTransform = t.uvTransform;
		// sheen roughness is in A
		out.fuzzRoughness.swizzle = 3;
	}
	out.fuzzColor.rgb = f32v3 {
		sh.sheen_color_factor[0],
		sh.sheen_color_factor[1],
		sh.sheen_color_factor[2],
	};
	out.fuzzRoughness.r = material_clamp_roughness(sh.sheen_roughness_factor);
	// gltf disables sheen via a black sheen color; a black openPBR fuzz layer
	// would still attenuate the base, so drive the weight from the color
	out.fuzzWeight.r = (
		std::max({
			sh.sheen_color_factor[0],
			sh.sheen_color_factor[1],
			sh.sheen_color_factor[2],
		})
	);
}

static void material_load_iridescence(
	[[maybe_unused]] ImplScene & s,
	[[maybe_unused]] cgltf_iridescence const & ir,
	[[maybe_unused]] GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_float iridescence_factor;
	// cgltf_texture_view iridescence_texture;
	// cgltf_float iridescence_ior;
	// cgltf_float iridescence_thickness_min;
	// cgltf_float iridescence_thickness_max;
	// cgltf_texture_view iridescence_thickness_texture;
	// PBR:
	// GpuMorMaterialComponent1 thinFilmWeight;
	// GpuMorMaterialComponent1 thinFilmThickness;
	// GpuMorMaterialComponent1 thinFilmIor;

	// the thickness texture lerps min->max in gltf but the material model only
	// multiplies, which is equivalent only when min = 0
	if (
		ir.iridescence_thickness_texture.texture
		&& ir.iridescence_thickness_min != 0.0f
	) {
		printf(
			"WARNING: iridescence thickness min %.1fnm ignored"
			" (texture scales 0 -> max)\n",
			ir.iridescence_thickness_min
		);
	}

	if (ir.iridescence_texture.texture) {
		TextureRef const t = load_texture(s, ir.iridescence_texture, false);
		out.thinFilmWeight.texture = t.handle;
		out.thinFilmWeight.uvTransform = t.uvTransform;
		// iridescence strength is in R
		out.thinFilmWeight.swizzle = 0;
	}
	if (ir.iridescence_thickness_texture.texture) {
		TextureRef const t = (
			load_texture(s, ir.iridescence_thickness_texture, false)
		);
		out.thinFilmThickness.texture = t.handle;
		out.thinFilmThickness.uvTransform = t.uvTransform;
		// iridescence thickness is in G
		out.thinFilmThickness.swizzle = 1;
	}
	// gltf iridescence thickness is in nanometers, thinFilmThickness is in
	// micrometers
	out.thinFilmThickness.r = ir.iridescence_thickness_max * 1e-3f;
	out.thinFilmWeight.r = ir.iridescence_factor;
	out.thinFilmIor.r = ir.iridescence_ior;
}

static void material_load_anisotropy(
	ImplScene & s,
	cgltf_material const & m,
	GpuMorMaterial & out
) {
	// GLTF (KHR_materials_anisotropy):
	// cgltf_float anisotropy_strength;
	// cgltf_float anisotropy_rotation;
	// cgltf_texture_view anisotropy_texture;
	// OpenPBR:
	// specularRoughnessAnisotropy;
	//
	// the extension modifies the base specular ndf, so it applies to every
	// pbr material; it is unrelated to KHR_materials_specular, and it is
	// not the henyey-greenstein transmissionScatterAnisotropy

	auto const & aniso = m.anisotropy;

	// khr and openpbr parameterize anisotropy differently (khr: alpha_t =
	// mix(r^2, 1, s^2), alpha_b = r^2; openpbr: alpha_t = alpha sqrt(2 / (1 +
	// (1 - a)^2)), alpha_b = (1 - a) alpha_t), but ratio-matching between them
	// blows up at low roughness -- a small r^2 makes alpha_b/alpha_t collapse
	// toward 0 for almost any nonzero strength, converting a to ~1 (the
	// numerically-floored, needle-thin axis) regardless of what was
	// authored. strength is already a similarly-shaped [0, 1] knob in both
	// systems, so just pass it through unconverted
	out.specularRoughnessAnisotropy.r = aniso.anisotropy_strength;

	if (aniso.anisotropy_texture.texture) {
		// direction is in RG (unused, see rotation TODO below), strength
		// multiplier is in B
		TextureRef const t = load_texture(s, aniso.anisotropy_texture, false);
		out.specularRoughnessAnisotropy.texture = t.handle;
		out.specularRoughnessAnisotropy.uvTransform = t.uvTransform;
		out.specularRoughnessAnisotropy.swizzle = 2;
	}

	// counter-clockwise from the tangent toward the bitangent, applied to
	// the specular shading frame via shadingFrameRotate
	out.specularRoughnessAnisotropyRotation.r = aniso.anisotropy_rotation;

	// TODO the RG direction channels of the anisotropy texture (per-pixel
	// direction) are still unused; decoding them means an angle texture
	// added onto this rotation at material load
}

static void material_load_dispersion(
	[[maybe_unused]] ImplScene & s,
	[[maybe_unused]] cgltf_dispersion const & disp,
	[[maybe_unused]] GpuMorMaterial & out
) {
	// GLTF:
	// cgltf_float dispersion;
	// OpenPBR:
	// transmissionDispersionScale;
	// transmissionDispersionAbbeNumber;

	// gltf defines dispersion = 20 / abbe with 0 meaning disabled
	if (disp.dispersion > 0.0f) {
		out.transmissionDispersionAbbeNumber.r = 20.0f / disp.dispersion;
		out.transmissionDispersionScale.r = 1.0f;
	}
}

static void load_primitive(
	ImplScene * s,
	cgltf_primitive const & prim,
	u32 const instanceIndex
) {
	if (prim.type != cgltf_primitive_type_triangles) return;

	cgltf_accessor * posAcc = nullptr;
	cgltf_accessor * normAcc = nullptr;
	cgltf_accessor * uvAcc = nullptr;
	cgltf_accessor * tanAcc = nullptr;
	for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai) {
		cgltf_attribute const & attr = prim.attributes[ai];
		switch (attr.type) {
			case cgltf_attribute_type_position: posAcc = attr.data; break;
			case cgltf_attribute_type_normal: normAcc = attr.data; break;
			case cgltf_attribute_type_texcoord:
				if (attr.index == 0) uvAcc = attr.data;
				break;
			case cgltf_attribute_type_tangent: tanAcc = attr.data; break;
			default: break;
		}
	}
	if (!posAcc) return;

	u32 const vertexBase = (u32)s->positions.size();
	u32 const vertexCount = (u32)posAcc->count;
	u32 const indexCount = (
		prim.indices ? (u32)prim.indices->count : vertexCount
	);

	// -- positions
	{
		std::vector<f32> tmp(vertexCount * 3);
		cgltf_accessor_unpack_floats(posAcc, tmp.data(), tmp.size());
		for (u32 i = 0; i < vertexCount; ++i) {
			s->positions.push_back({ tmp[i*3+0], tmp[i*3+1], tmp[i*3+2] });
		}
	}

	// -- normals, uvs, tangents
	{
		std::vector<f32> normals(vertexCount * 3, 0.0f);
		std::vector<f32> uvs(vertexCount * 2, 0.0f);
		std::vector<f32> tangents(vertexCount * 4, 0.0f);
		if (normAcc) {
			cgltf_accessor_unpack_floats(normAcc, normals.data(), normals.size());
		}
		if (uvAcc) {
			cgltf_accessor_unpack_floats(uvAcc, uvs.data(), uvs.size());
		}
		if (tanAcc) {
			cgltf_accessor_unpack_floats(tanAcc, tangents.data(), tangents.size());
		} else {
			for (u32 i = 0; i < vertexCount; ++i) {
				tangents[i*4+0] = 1.0f;
				tangents[i*4+3] = 1.0f;
			}
		}
		for (u32 i = 0; i < vertexCount; ++i) {
			s->attributes.push_back({
				.normal = { normals[i*3+0], normals[i*3+1], normals[i*3+2] },
				.uv = { uvs[i*2+0], uvs[i*2+1] },
				.tangent = {
					tangents[i*4+0], tangents[i*4+1],
					tangents[i*4+2], tangents[i*4+3],
				},
			});
		}
	}

	// -- indices
	std::vector<u32> indices(indexCount);
	if (prim.indices) {
		for (u32 i = 0; i < indexCount; ++i) {
			indices[i] = (u32)cgltf_accessor_read_index(prim.indices, i);
		}
	} else {
		for (u32 i = 0; i < indexCount; ++i) {
			indices[i] = i;
		}
	}

	// -- materials
	u32 materialIndex = 0;
	if (prim.material) {
		auto const it = s->materialIndices.find(prim.material);
		if (it != s->materialIndices.end()) {
			materialIndex = it->second;
		} else {
			materialIndex = (u32)s->materials.size();
			cgltf_material const & mat = *prim.material;
			GpuMorMaterial gpuMaterial = material_load_default();

			// spec-gloss wins over the core block when both are present:
			// the extension is the authored intent, the core block is the
			// compatibility fallback
			if (mat.has_pbr_specular_glossiness) {
				material_load_pbr_specular_glossiness(
					*s, mat.pbr_specular_glossiness, gpuMaterial
				);
			} else if (mat.has_pbr_metallic_roughness)  {
				material_load_pbr_metallic_roughness(
					*s, mat.pbr_metallic_roughness, gpuMaterial
				);
			}
			if (mat.has_clearcoat) {
				material_load_clearcoat(*s, mat.clearcoat, gpuMaterial);
			}
			if (mat.has_transmission) {
				material_load_transmission(*s, mat.transmission, gpuMaterial);
			}
			if (mat.has_diffuse_transmission) {
				material_load_diffuse_transmission(
					*s, mat.diffuse_transmission, gpuMaterial
				);
			}
			if (mat.has_volume) {
				material_load_volume(*s, mat.volume, gpuMaterial);
			}
			// khr_materials_transmission without khr_materials_volume (or
			// with an explicit zero thickness) is an infinitesimally thin
			// shell per spec: there is no second surface to refract back
			// out of, so transmission passes straight through undeviated
			// instead of bending like a solid volume. khr_materials_-
			// diffuse_transmission takes the same rule, and its thin case
			// is the one the openpbr sheet lobe models
			if (mat.has_transmission || mat.has_diffuse_transmission) {
				bool const thinWalled = (
					!mat.has_volume || mat.volume.thickness_factor <= 0.0f
				);
				gpuMaterial.geometryThinWalled.r = thinWalled ? 1.0f : 0.0f;
				// a diffuse-transmissive material with a real interior runs
				// the subsurface walk instead of the sheet, so it needs the
				// volume's attenuation as a mean free path
				if (mat.has_diffuse_transmission && !thinWalled) {
					material_load_diffuse_transmission_volume(
						*s, mat, gpuMaterial
					);
				}
			}
			if (mat.has_ior) {
				material_load_ior(*s, mat, gpuMaterial);
			}
			if (mat.has_specular) {
				material_load_specular(*s, mat.specular, gpuMaterial);
			}
			if (mat.has_sheen) {
				material_load_sheen(*s, mat.sheen, gpuMaterial);
			}
			if (mat.has_emissive_strength) {
				material_load_emission(*s, mat.emissive_strength, gpuMaterial);
			}
			if (mat.has_iridescence) {
				material_load_iridescence(*s, mat.iridescence, gpuMaterial);
			}
			if (mat.has_anisotropy) {
				material_load_anisotropy(*s, mat, gpuMaterial);
			}
			if (mat.has_dispersion) {
				material_load_dispersion(*s, mat.dispersion, gpuMaterial);
			}

			// -- load in non-optional parameters
			{
				TextureRef const t = load_texture(*s, mat.normal_texture, false);
				gpuMaterial.geometryNormalTexture = t.handle;
				gpuMaterial.geometryNormalUvTransform = t.uvTransform;
			}
			{
				TextureRef const t = load_texture(*s, mat.emissive_texture, true);
				gpuMaterial.emissionColor.texture = t.handle;
				gpuMaterial.emissionColor.uvTransform = t.uvTransform;
			}
			gpuMaterial.emissionColor.rgb = f32v3 {
				mat.emissive_factor[0],
				mat.emissive_factor[1],
				mat.emissive_factor[2],
			};
			// core gltf emissive has no strength extension; a non-zero
			// emissive factor implies unit luminance
			if (
				!mat.has_emissive_strength
				&& (
					mat.emissive_factor[0] != 0.0f
					|| mat.emissive_factor[1] != 0.0f
					|| mat.emissive_factor[2] != 0.0f
				)
			) {
				gpuMaterial.emissionLuminance.r = 1.0f;
			}
			if (mat.alpha_mode == cgltf_alpha_mode_mask) {
				gpuMaterial.alphaCutoff = mat.alpha_cutoff;
			} else {
				gpuMaterial.alphaCutoff = 0.0f;
			}
			// opaque alpha mode ignores the base color alpha entirely
			if (mat.alpha_mode == cgltf_alpha_mode_opaque) {
				gpuMaterial.geometryOpacity.texture = 0u;
				gpuMaterial.geometryOpacity.r = 1.0f;
				gpuMaterial.geometryOpacity.swizzle = 0;
			} else {
				s->hasNonOpaqueMaterial = true;
			}
			// TODO double_sided, unlit
			s->materialIndices.emplace(prim.material, materialIndex);
			s->materials.push_back(gpuMaterial);
			s->materialNames.emplace_back(mat.name ? mat.name : "");
		}
	}

	// -- meshlets
	size_t const maxMeshlets = meshopt_buildMeshletsBound(
		indexCount, skMaxMeshletVerts, skMaxMeshletTris
	);
	std::vector<meshopt_Meshlet> optMeshlets(maxMeshlets);
	std::vector<u32> optVerts(maxMeshlets * skMaxMeshletVerts);
	std::vector<u8> optTris(maxMeshlets * skMaxMeshletTris * 3);

	u32 const primMeshletCount = (
		(u32)meshopt_buildMeshlets(
			optMeshlets.data(), optVerts.data(), optTris.data(),
			indices.data(), indexCount,
			reinterpret_cast<f32 const *>(s->positions.data() + vertexBase),
			vertexCount, sizeof(f32v3),
			skMaxMeshletVerts, skMaxMeshletTris, 0.0f
		)
	);

	// trim to actual used extents
	meshopt_Meshlet const & last = optMeshlets[primMeshletCount - 1];
	optVerts.resize(last.vertex_offset + last.vertex_count);
	optTris.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3u));

	u32 const vertOffsetBase = (u32)s->meshletVerts.size();
	u32 const triOffsetBase = (u32)s->meshletTris.size();

	for (u32 const v : optVerts) {
		s->meshletVerts.push_back(vertexBase + v);
	}
	for (u8 const t : optTris) {
		s->meshletTris.push_back(t);
	}

	for (u32 mi = 0; mi < primMeshletCount; ++mi) {
		meshopt_Meshlet const & m = optMeshlets[mi];
		s->meshlets.push_back({
			.vertexOffset = vertOffsetBase + m.vertex_offset,
			.vertexCount = m.vertex_count,
			.triangleOffset = triOffsetBase + m.triangle_offset,
			.triangleCount = m.triangle_count,
			.instanceIndex = instanceIndex,
			.materialIndex = (u32)materialIndex,
		});
	}
}

static void load_node(ImplScene * s, cgltf_node const * node) {
	if (node->mesh) {
		u32 const instanceIndex = (u32)s->instances.size();
		u32 const meshletOffset = (u32)s->meshlets.size();
		u32 const vertexBase = (u32)s->positions.size();

		for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
			load_primitive(s, node->mesh->primitives[pi], instanceIndex);
		}

		f32m44 transform {};
		cgltf_node_transform_world(node, transform.m.ptr());

		// bake node world transform into positions and attributes so the BLAS
		// lives in GLTF-world space; instance.transform is then identity and
		// the TLAS only needs the scene-level make_model_matrix.
		f32 const * m = transform.m.ptr();
		f32v3 const col0 { m[0], m[1], m[2] };
		f32v3 const col1 { m[4], m[5], m[6] };
		f32v3 const col2 { m[8], m[9], m[10] };
		// tangents follow surface directions and transform by the 3x3 directly
		auto const lin3 = [&](f32v3 const v) -> f32v3 {
			return {
				col0.x*v.x + col1.x*v.y + col2.x*v.z,
				col0.y*v.x + col1.y*v.y + col2.y*v.z,
				col0.z*v.x + col1.z*v.y + col2.z*v.z,
			};
		};
		// normals are covectors and transform by the cofactor matrix
		// det(M) M^{-T} = [c1 x c2, c2 x c0, c0 x c1], which stays
		// perpendicular to the surface under non-uniform scale
		f32v3 const cof0 = f32v3_cross(col1, col2);
		f32v3 const cof1 = f32v3_cross(col2, col0);
		f32v3 const cof2 = f32v3_cross(col0, col1);
		auto const cof3 = [&](f32v3 const v) -> f32v3 {
			return {
				cof0.x*v.x + cof1.x*v.y + cof2.x*v.z,
				cof0.y*v.x + cof1.y*v.y + cof2.y*v.z,
				cof0.z*v.x + cof1.z*v.y + cof2.z*v.z,
			};
		};
		auto const norm = [](f32v3 const v) -> f32v3 {
			f32 const inv = 1.0f / sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
			return { v.x*inv, v.y*inv, v.z*inv };
		};

		u32 const vertexEnd = (u32)s->positions.size();
		for (u32 i = vertexBase; i < vertexEnd; ++i) {
			f32v3 const p = s->positions[i];
			f32v4 const tp = transform * f32v4 { p.x, p.y, p.z, 1.0f };
			s->positions[i] = { tp.x, tp.y, tp.z };

			GpuMorVertexAttribute & attr = s->attributes[i];
			attr.normal = norm(cof3(attr.normal));
			f32v3 const rt = (
				norm(lin3({ attr.tangent.x, attr.tangent.y, attr.tangent.z }))
			);
			attr.tangent = { rt.x, rt.y, rt.z, attr.tangent.w };
		}

		s->instances.push_back({
			.transform = f32m44_identity(),
			.meshletOffset = meshletOffset,
			.meshletCount = (u32)s->meshlets.size() - meshletOffset,
		});
	}

	for (cgltf_size ci = 0; ci < node->children_count; ++ci) {
		load_node(s, node->children[ci]);
	}
}

static vkof::Buffer upload_buffer(void const * data, u64 byteCount) {
	SRAT_ASSERT_ALWAYS(byteCount > 0);
	vkof::Buffer const buf = vkof::buffer_create({
		.byteCount = byteCount,
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_upload({
		.buffer = buf,
		.byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(data), byteCount
		),
	});
	return buf;
}

// ----------------------------------------------------------------------------
// -- public API
// ----------------------------------------------------------------------------

mor::Scene mor::scene_create() {
	ImplScene * const s = new ImplScene();
	// index 0 is always the default material; meshlets with no material use it
	s->materials.push_back(material_load_default());
	s->materialNames.push_back("");
	return mor::Scene { .id = reinterpret_cast<u64>(s) };
}

void mor::scene_destroy(mor::Scene const & scene) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);
	for (ImTextureID const id : s->imguiIds) {
		if (id) { vkof::image_imgui_id_destroy(id); }
	}
	for (vkof::Image const & img : s->images) {
		vkof::image_destroy(img);
	}
	delete s;
}

void mor::sampler_cache_destroy() {
	for (auto const & [key, sampler] : sSamplerCache) {
		vkof::sampler_destroy(sampler);
	}
	sSamplerCache.clear();
	if (sImguiDisplaySampler.id) {
		vkof::sampler_destroy(sImguiDisplaySampler);
		sImguiDisplaySampler = { 0u };
	}
}

void mor::scene_set_anisotropy(
	Scene const & scene,
	GpuScene const & gpuScene,
	f32 const & anisotropy
) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);
	ImplGpuScene * const gpu = reinterpret_cast<ImplGpuScene *>(gpuScene.id);

	for (auto const & [key, sampler] : sSamplerCache) {
		vkof::sampler_destroy(sampler);
	}
	sSamplerCache.clear();

	std::unordered_map<u32, u32> & handleRemap = s->lastHandleRemap;
	handleRemap.clear();
	for (auto & [tex, oldHandle] : s->textureHandles) {
		SamplerKey key = s->textureSamplerKeys.at(tex);
		key.maxAnisotropy = anisotropy;

		auto cacheIt = sSamplerCache.find(key);
		if (cacheIt == sSamplerCache.end()) {
			vkof::Sampler const sampler = vkof::sampler_create({
				.magFilter = key.magFilter,
				.minFilter = key.minFilter,
				.addressModeU = key.addressU,
				.addressModeV = key.addressV,
				.addressModeW = vkof::SamplerAddressMode::repeat,
				.mipmapMode = key.mipmapMode,
				.maxAnisotropy = key.maxAnisotropy,
			});
			cacheIt = sSamplerCache.emplace(key, sampler).first;
		}

		u32 const newHandle = vkof::image_sampler_handle({
			.image = s->textureImages.at(tex),
			.sampler = cacheIt->second,
		});
		handleRemap[oldHandle] = newHandle;
		oldHandle = newHandle;
	}

	for (GpuMorMaterial & mat : s->materials) {
		auto const remap = [&](u32 h) -> u32 {
			if (h == 0u) { return 0u; }
			auto const it = handleRemap.find(h);
			return it != handleRemap.end() ? it->second : h;
		};
		#define REMAP_MATERIAL_TEXTURE(type, field, min, max) \
			mat. field .texture = remap( mat. field .texture );
		MOR_MATERIAL_ALL_PARAMS(REMAP_MATERIAL_TEXTURE)
		#undef REMAP_MATERIAL_TEXTURE
		#define REMAP_NORMAL_TEXTURE(field) \
			mat. field = remap( mat. field );
		MOR_MATERIAL_ALL_TEXTURES_NORMAL(REMAP_NORMAL_TEXTURE)
		#undef REMAP_NORMAL_TEXTURE
	}

	if (!s->materials.empty()) {
		vkof::buffer_upload({
			.buffer = gpu->materials,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(s->materials.data()),
				s->materials.size() * sizeof(GpuMorMaterial)
			),
		});
	}
}

u32 mor::scene_instance_count(mor::Scene const & scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene.id)->instances.size();
}

u32 mor::scene_meshlet_count(mor::Scene const & scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene.id)->meshlets.size();
}

u32 mor::scene_vertex_count(mor::Scene const & scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene.id)->positions.size();
}

void mor::scene_bounds(mor::Scene const & scene, f32v3 & outMin, f32v3 & outMax) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	outMin = { FLT_MAX, FLT_MAX, FLT_MAX };
	outMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	for (f32v3 const & p : s->positions) {
		outMin.x = std::min(outMin.x, p.x);
		outMin.y = std::min(outMin.y, p.y);
		outMin.z = std::min(outMin.z, p.z);
		outMax.x = std::max(outMax.x, p.x);
		outMax.y = std::max(outMax.y, p.y);
		outMax.z = std::max(outMax.z, p.z);
	}
}

bool mor::scene_is_fully_opaque(mor::Scene const & scene) {
	return !reinterpret_cast<ImplScene const *>(scene.id)->hasNonOpaqueMaterial;
}

void mor::scene_imgui_debug(mor::Scene const & scene) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);

	u32 const instanceCount = (u32)s->instances.size();
	u32 const meshletCount = (u32)s->meshlets.size();
	u32 const vertexCount = (u32)s->positions.size();
	u32 const materialCount = (u32)s->materials.size();
	u32 const textureCount = (u32)s->images.size();

	char label[64];

	snprintf(label, sizeof(label), "instances (%u)", instanceCount);
	if (ImGui::TreeNode(label)) {
		for (u32 i = 0; i < instanceCount; ++i) {
			GpuMorInstance const & inst = s->instances[i];
			ImGui::Text(
				"[%u] meshlets: %u  offset: %u",
				i, inst.meshletCount, inst.meshletOffset
			);
		}
		ImGui::TreePop();
	}

	snprintf(label, sizeof(label), "meshlets (%u)", meshletCount);
	if (ImGui::TreeNode(label)) {
		for (u32 i = 0; i < meshletCount; ++i) {
			GpuMorMeshlet const & m = s->meshlets[i];
			ImGui::Text(
				"[%u] verts: %u  tris: %u  inst: %u  mat: %u",
				i, m.vertexCount, m.triangleCount, m.instanceIndex, m.materialIndex
			);
		}
		ImGui::TreePop();
	}

	ImGui::Text("vertices:  %u", vertexCount);

	snprintf(label, sizeof(label), "textures (%u)", textureCount);
	if (ImGui::TreeNode(label)) {
		u32 i = 0u;
		for (auto const & [tex, handle] : s->textureHandles) {
			ImGui::Text("[%u] handle: %u", i, handle);
			++i;
		}
		ImGui::TreePop();
	}
}

void mor::scene_imgui_textures(mor::Scene const & scene) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);
	if (s->images.empty()) { return; }
	if (!sImguiDisplaySampler.id) {
		sImguiDisplaySampler = vkof::sampler_create({
			.magFilter = vkof::SamplerFilter::linear,
			.minFilter = vkof::SamplerFilter::linear,
			.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
			.mipmapMode = vkof::SamplerMipmapMode::linear,
			.maxAnisotropy = 1.0f,
		});
	}
	if (s->imguiIds.empty()) {
		s->imguiIds.resize(s->images.size(), ImTextureID(0));
		for (u32 i = 0u; i < (u32)s->images.size(); ++i) {
			s->imguiIds[i] = vkof::image_imgui_id({
				.image = s->images[i],
				.sampler = sImguiDisplaySampler,
			});
		}
	}
	constexpr f32 skThumbSize = 64.0f;
	for (u32 i = 0u; i < (u32)s->imguiIds.size(); ++i) {
		char const * const name = (
			i < (u32)s->imageNames.size()
			? s->imageNames[i].c_str()
			: "[unknown]"
		);
		ImGui::TextUnformatted(name);
		ImGui::Image(s->imguiIds[i], ImVec2(skThumbSize, skThumbSize));
	}
}

void mor::scene_load_gltf(mor::Scene const & scene, char const * const path) {
	ImplScene * const s = reinterpret_cast<ImplScene *>(scene.id);

	s->gltfDir = std::filesystem::path(path).parent_path().string();

	cgltf_options const options {};
	cgltf_data * data = nullptr;
	SRAT_ASSERT_ALWAYS(
		cgltf_parse_file(&options, path, &data) == cgltf_result_success
	);
	SRAT_ASSERT_ALWAYS(
		cgltf_load_buffers(&options, data, path) == cgltf_result_success
	);

	s->data = data;
	for (cgltf_size si = 0; si < data->scenes_count; ++si) {
		cgltf_scene const & gltfScene = data->scenes[si];
		for (cgltf_size ni = 0; ni < gltfScene.nodes_count; ++ni) {
			load_node(s, gltfScene.nodes[ni]);
		}
	}
	s->data = nullptr;

	cgltf_free(data);
}

mor::GpuScene mor::scene_gpu_upload(mor::Scene const & scene) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	ImplGpuScene * gpu = new ImplGpuScene();

	gpu->positions = (
		upload_buffer(
			s->positions.data(), s->positions.size() * sizeof(f32v3)
		)
	);
	gpu->attributes = (
		upload_buffer(
			s->attributes.data(), s->attributes.size() * sizeof(GpuMorVertexAttribute)
		)
	);
	gpu->meshletVerts = (
		upload_buffer(
			s->meshletVerts.data(), s->meshletVerts.size() * sizeof(u32)
		)
	);
	gpu->meshletTris = (
		upload_buffer(
			s->meshletTris.data(), s->meshletTris.size() * sizeof(u8)
		)
	);
	gpu->meshlets = (
		upload_buffer(
			s->meshlets.data(), s->meshlets.size() * sizeof(GpuMorMeshlet)
		)
	);
	gpu->instances = (
		upload_buffer(
			s->instances.data(), s->instances.size() * sizeof(GpuMorInstance)
		)
	);
	if (!s->materials.empty()) {
		gpu->materials = (
			upload_buffer(
				s->materials.data(), s->materials.size() * sizeof(GpuMorMaterial)
			)
		);
	}
	if (!s->uvTransforms.empty()) {
		gpu->uvTransforms = (
			upload_buffer(
				s->uvTransforms.data(),
				s->uvTransforms.size() * sizeof(GpuMorUvTransform)
			)
		);
	}
	gpu->meshletCount = (u32)s->meshlets.size();
	gpu->vertexCount = (u32)s->positions.size();

	// -- flat u32 index buffer for BLAS geometry + parallel meshlet index per triangle
	std::vector<u32> flatIndices;
	std::vector<u32> flatMeshlets;
	for (u32 mi = 0u; mi < (u32)s->meshlets.size(); ++mi) {
		GpuMorMeshlet const & m = s->meshlets[mi];
		for (u32 tri = 0u; tri < m.triangleCount; ++tri) {
			u32 const base = m.triangleOffset + tri * 3u;
			flatIndices.emplace_back(
				s->meshletVerts[m.vertexOffset + s->meshletTris[base + 0u]]
			);
			flatIndices.emplace_back(
				s->meshletVerts[m.vertexOffset + s->meshletTris[base + 1u]]
			);
			flatIndices.emplace_back(
				s->meshletVerts[m.vertexOffset + s->meshletTris[base + 2u]]
			);
			flatMeshlets.emplace_back(mi);
		}
	}
	gpu->triangleCount = (u32)(flatIndices.size() / 3u);
	gpu->flatIndices = (
		upload_buffer(
			flatIndices.data(), flatIndices.size() * sizeof(u32)
		)
	);
	gpu->flatMeshlets = (
		upload_buffer(
			flatMeshlets.data(), flatMeshlets.size() * sizeof(u32)
		)
	);

	return mor::GpuScene { .id = reinterpret_cast<u64>(gpu) };
}

void mor::scene_gpu_destroy(mor::GpuScene const & scene) {
	ImplGpuScene * const gpu = reinterpret_cast<ImplGpuScene *>(scene.id);
	vkof::buffer_destroy(gpu->positions);
	vkof::buffer_destroy(gpu->attributes);
	vkof::buffer_destroy(gpu->meshletVerts);
	vkof::buffer_destroy(gpu->meshletTris);
	vkof::buffer_destroy(gpu->meshlets);
	vkof::buffer_destroy(gpu->instances);
	vkof::buffer_destroy(gpu->materials);
	vkof::buffer_destroy(gpu->uvTransforms);
	vkof::buffer_destroy(gpu->textures);
	vkof::buffer_destroy(gpu->flatIndices);
	vkof::buffer_destroy(gpu->flatMeshlets);
	delete gpu;
}

mor::Buffers mor::scene_gpu_buffers(mor::GpuScene const & scene) {
	ImplGpuScene const * const gpu = reinterpret_cast<ImplGpuScene const *>(scene.id);
	return {
		.meshlets = vkof::buffer_virtual_address(gpu->meshlets),
		.materials = vkof::buffer_virtual_address(gpu->materials),
		.uvTransforms = vkof::buffer_virtual_address(gpu->uvTransforms),
		.textures = vkof::buffer_virtual_address(gpu->textures),
		.instances = vkof::buffer_virtual_address(gpu->instances),
		.positions = vkof::buffer_virtual_address(gpu->positions),
		.attributes = vkof::buffer_virtual_address(gpu->attributes),
		.meshletVerts = vkof::buffer_virtual_address(gpu->meshletVerts),
		.meshletTris = vkof::buffer_virtual_address(gpu->meshletTris),
		.flatIndices = vkof::buffer_virtual_address(gpu->flatIndices),
		.flatMeshlets = vkof::buffer_virtual_address(gpu->flatMeshlets),
		.vertexCount = gpu->vertexCount,
		.triangleCount = gpu->triangleCount,
	};
}

u32 mor::scene_gpu_meshlet_count(mor::GpuScene const & scene) {
	return reinterpret_cast<ImplGpuScene const *>(scene.id)->meshletCount;
}

u32 mor::scene_material_count(mor::Scene const & scene) {
	return (u32)reinterpret_cast<ImplScene const *>(scene.id)->materials.size();
}

std::string mor::scene_material_name(mor::Scene const & scene, u32 const index) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	SRAT_ASSERT(index < (u32)s->materialNames.size());
	return s->materialNames[index];
}

GpuMorMaterial mor::scene_material_get(mor::Scene const & scene, u32 const index) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	SRAT_ASSERT(index < (u32)s->materials.size());
	return s->materials[index];
}

std::vector<mor::MeshletAreaInfo> mor::scene_meshlet_area_info(
	mor::Scene const & scene
) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	std::vector<mor::MeshletAreaInfo> info;
	info.reserve(s->meshlets.size());
	for (GpuMorMeshlet const & m : s->meshlets) {
		f32 area = 0.0f;
		for (u32 tri = 0u; tri < m.triangleCount; ++tri) {
			u32 const triBase = m.triangleOffset + tri * 3u;
			u32 const local0 = (u32)s->meshletTris[triBase + 0u];
			u32 const local1 = (u32)s->meshletTris[triBase + 1u];
			u32 const local2 = (u32)s->meshletTris[triBase + 2u];
			f32v3 const pos0 = (
				s->positions[s->meshletVerts[m.vertexOffset + local0]]
			);
			f32v3 const pos1 = (
				s->positions[s->meshletVerts[m.vertexOffset + local1]]
			);
			f32v3 const pos2 = (
				s->positions[s->meshletVerts[m.vertexOffset + local2]]
			);
			area += (
				0.5f * f32v3_length(f32v3_cross(pos1 - pos0, pos2 - pos0))
			);
		}
		info.emplace_back(mor::MeshletAreaInfo {
			.materialIndex = m.materialIndex,
			.area = area,
			.triangleCount = m.triangleCount,
		});
	}
	return info;
}

mor::GpuMaterials mor::scene_gpu_materials_create(mor::Scene const & scene) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	ImplGpuMaterials * const m = new ImplGpuMaterials();
	m->cpu = s->materials;
	if (!m->cpu.empty()) {
		m->buffer = upload_buffer(
			m->cpu.data(), m->cpu.size() * sizeof(GpuMorMaterial)
		);
	}
	return mor::GpuMaterials { .id = reinterpret_cast<u64>(m) };
}

void mor::scene_gpu_materials_destroy(mor::GpuMaterials const & mats) {
	ImplGpuMaterials * const m = reinterpret_cast<ImplGpuMaterials *>(mats.id);
	vkof::buffer_destroy(m->buffer);
	delete m;
}

u64 mor::scene_gpu_materials_va(mor::GpuMaterials const & mats) {
	ImplGpuMaterials const * const m = (
		reinterpret_cast<ImplGpuMaterials const *>(mats.id)
	);
	return vkof::buffer_virtual_address(m->buffer);
}

u32 mor::scene_gpu_materials_count(mor::GpuMaterials const & mats) {
	ImplGpuMaterials const * const m = (
		reinterpret_cast<ImplGpuMaterials const *>(mats.id)
	);
	return (u32)m->cpu.size();
}

GpuMorMaterial mor::scene_gpu_materials_get(
	mor::GpuMaterials const & mats, u32 const index
) {
	ImplGpuMaterials const * const m = (
		reinterpret_cast<ImplGpuMaterials const *>(mats.id)
	);
	SRAT_ASSERT(index < (u32)m->cpu.size());
	return m->cpu[index];
}

void mor::scene_material_override_scalars(
	mor::GpuMaterials const & mats, u32 const index, GpuMorMaterial const & scalars
) {
	ImplGpuMaterials * const m = reinterpret_cast<ImplGpuMaterials *>(mats.id);
	SRAT_ASSERT(index < (u32)m->cpu.size());
	// the scene desc can disagree with the loaded model on material count
	if (index >= (u32)m->cpu.size()) {
		return;
	}
	GpuMorMaterial & dst = m->cpu[index];
	#define COPY_FIELD(type, name, min, max) \
		dst.name = scalars.name;
	MOR_MATERIAL_ALL_PARAMS(COPY_FIELD)
	#undef COPY_FIELD
}

void mor::scene_gpu_materials_sync_textures(
	mor::Scene const & scene, mor::GpuMaterials const & mats
) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	ImplGpuMaterials * const m = reinterpret_cast<ImplGpuMaterials *>(mats.id);
	u32 const count = (u32)std::min(m->cpu.size(), s->materials.size());
	for (u32 i = 0u; i < count; ++i) {
		GpuMorMaterial & dst = m->cpu[i];
		GpuMorMaterial const & src = s->materials[i];
		#define COPY_FIELD(type, name, min, max) \
			dst.name.texture = src.name.texture;
		MOR_MATERIAL_ALL_PARAMS(COPY_FIELD)
		#undef COPY_FIELD
		#define COPY_NORMAL_TEXTURE(name) \
			dst.name = src.name;
		MOR_MATERIAL_ALL_TEXTURES_NORMAL(COPY_NORMAL_TEXTURE)
		#undef COPY_NORMAL_TEXTURE
	}
}

void mor::scene_gpu_materials_upload(mor::GpuMaterials const & mats) {
	ImplGpuMaterials const * const m = (
		reinterpret_cast<ImplGpuMaterials const *>(mats.id)
	);
	if (m->cpu.empty()) {
		return;
	}
	vkof::buffer_upload({
		.buffer = m->buffer,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(m->cpu.data()),
			m->cpu.size() * sizeof(GpuMorMaterial)
		),
	});
}

void mor::scene_remap_material_textures(
	mor::Scene const & scene, GpuMorMaterial & mat
) {
	ImplScene const * const s = reinterpret_cast<ImplScene const *>(scene.id);
	auto const remap = [&](u32 const h) -> u32 {
		if (h == 0u) { return 0u; }
		auto const it = s->lastHandleRemap.find(h);
		return it != s->lastHandleRemap.end() ? it->second : h;
	};
	#define REMAP_MATERIAL_TEXTURE(type, field, min, max) \
		mat. field .texture = remap(mat. field .texture);
	MOR_MATERIAL_ALL_PARAMS(REMAP_MATERIAL_TEXTURE)
	#undef REMAP_MATERIAL_TEXTURE
	#define REMAP_NORMAL_TEXTURE(field) \
		mat. field = remap(mat. field);
	MOR_MATERIAL_ALL_TEXTURES_NORMAL(REMAP_NORMAL_TEXTURE)
	#undef REMAP_NORMAL_TEXTURE
}
