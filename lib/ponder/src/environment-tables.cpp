#include <ponder/environment-tables.hpp>

#include <tinyexr.h>

#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------------
// -- private api
// -----------------------------------------------------------------------------

// computes euirect solid-angle Jacobian-weighted luminance importance per texel
void compute_luminance_weighted_importance(
	std::vector<float> & importance,
	const float * pixels,
	int32_t const width,
	int32_t const height
) {
	importance.resize(width * height);
	for (int32_t y = 0; y < height; ++y) {
		// theta matches to texel's center
		const float theta = (y + 0.5f) / height * M_PI;
		const float sinTheta = std::sin(theta);
		for (int32_t x = 0; x < width; ++x) {
			const int32_t idx = (y * width + x) * 4;
			const float r = pixels[idx + 0];
			const float g = pixels[idx + 1];
			const float b = pixels[idx + 2];
			const float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
			importance[y * width + x] = luminance * sinTheta;
		}
	}
}

// Vose's alias method: builds an O(1)-samplable discrete distribution over
// importance's texels (replaces a marginal+conditional cdf binary search --
// see environmentMapImportanceSample). totalSum <= 0 (all-black map, or a
// NaN-contaminated sum -- NaN > 0.0f is false) falls back to a uniform
// distribution over every texel, matching the old cdf builders' identical
// fallback for the same conditions
void compute_alias_table(
	std::vector<float> & prob,
	std::vector<uint32_t> & alias,
	std::vector<float> const & importance,
	float const totalSum
) {
	size_t const n = importance.size();
	prob.resize(n);
	alias.resize(n);
	std::vector<float> scaled(n);
	for (size_t i = 0; i < n; ++i) {
		scaled[i] = (
			totalSum > 0.0f
				? importance[i] / totalSum * (float)n
				: 1.0f
		);
	}
	std::vector<uint32_t> small, large;
	small.reserve(n);
	large.reserve(n);
	for (size_t i = 0; i < n; ++i) {
		(scaled[i] < 1.0f ? small : large).push_back((uint32_t)i);
	}
	while (!small.empty() && !large.empty()) {
		uint32_t const l = small.back();
		small.pop_back();
		uint32_t const g = large.back();
		large.pop_back();
		prob[l] = scaled[l];
		alias[l] = g;
		scaled[g] = (scaled[g] + scaled[l]) - 1.0f;
		(scaled[g] < 1.0f ? small : large).push_back(g);
	}
	// leftover entries (rounding fuzz put them all on one side): exact 1.0
	// probability, self-aliased
	while (!large.empty()) {
		uint32_t const g = large.back();
		large.pop_back();
		prob[g] = 1.0f;
		alias[g] = g;
	}
	while (!small.empty()) {
		uint32_t const l = small.back();
		small.pop_back();
		prob[l] = 1.0f;
		alias[l] = l;
	}
}

// computes normalized pdf from importance and total sum
void compute_pdf(
	std::vector<float> & pdf,
	std::vector<float> const & importance,
	f32 const totalSum,
	int32_t const width,
	int32_t const height
) {
	pdf.resize(width * height);
	for (i32 i = 0; i < width*height; ++i) {
		pdf[i] = (totalSum > 0.0f) ? importance[i] / totalSum : 0.0f;
	}
}

// -----------------------------------------------------------------------------
// -- public api
// -----------------------------------------------------------------------------

ponder::EnvironmentTables ponder::environment_tables_create(
	const std::string & path
) {
	float * pixels = nullptr;
	int32_t width = 0;
	int32_t height = 0;
	char const * err = nullptr;
	int32_t const ret = LoadEXR(&pixels, &width, &height, path.c_str(), &err);
	if (ret != TINYEXR_SUCCESS) {
		printf(
			"WARNING: failed to load environment map '%s': %s\n",
			path.c_str(), err ? err : "unknown error"
		);
		if (err) {
			FreeEXRErrorMessage(err);
		}
		return {};
	}

	// -- luminance-weighted importance
	std::vector<float> importance;
	compute_luminance_weighted_importance(
		importance,
		pixels,
		width,
		height
	);

	f32 totalSum = 0.0f;
	for (float const w : importance) {
		totalSum += w;
	}

	// -- pdf
	std::vector<float> pdf;
	compute_pdf(
		pdf,
		importance,
		totalSum,
		width,
		height
	);

	// -- alias table
	std::vector<float> aliasProb;
	std::vector<uint32_t> aliasIndexU32;
	compute_alias_table(aliasProb, aliasIndexU32, importance, totalSum);
	std::vector<float> aliasIndex(aliasIndexU32.size());
	for (size_t i = 0; i < aliasIndexU32.size(); ++i) {
		aliasIndex[i] = (float)aliasIndexU32[i];
	}

	// mip chain lets prefilter passes (env-prefilter-specular.comp etc) read
	// a pre-blurred source instead of point-sampling individual hdri
	// texels -- point-sampling a raw sun disc either hits its full peak
	// brightness or misses it entirely, no sample count converges that
	u32 const radianceMipCount = (
		(u32)std::floor(std::log2((f32)std::max(width, height))) + 1u
	);

	// -- upload to GPU
	EnvironmentTables tables {
		.radianceSampler = vkof::sampler_create({
			.magFilter = vkof::SamplerFilter::linear,
			.minFilter = vkof::SamplerFilter::linear,
			.addressModeU = vkof::SamplerAddressMode::repeat,
			.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
			.mipmapMode = vkof::SamplerMipmapMode::linear,
		}),
		.tableSampler = vkof::sampler_create({
			.magFilter = vkof::SamplerFilter::nearest,
			.minFilter = vkof::SamplerFilter::nearest,
			.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
			.mipmapMode = vkof::SamplerMipmapMode::nearest,
		}),
		.radianceImage = (
			vkof::image_create({
				.width = (u32)width,
				.height = (u32)height,
				.depth = 1u,
				.format = vkof::ImageFormat::r32g32b32a32_sfloat,
				.mipLevels = radianceMipCount,
				.optInitialData = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(pixels),
					sizeof(f32) * width * height * 4
				),
			})
		),
		.pdfImage = (
			vkof::image_create({
				.width = (u32)width,
				.height = (u32)height,
				.depth = 1u,
				.format = vkof::ImageFormat::r32_float,
				.mipLevels = 1u,
				.optInitialData = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(pdf.data()),
					sizeof(f32) * width * height
				),
			})
		),
		.aliasProbImage = (
			vkof::image_create({
				.width = (u32)width,
				.height = (u32)height,
				.depth = 1u,
				.format = vkof::ImageFormat::r32_float,
				.mipLevels = 1u,
				.optInitialData = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(aliasProb.data()),
					sizeof(f32) * width * height
				),
			})
		),
		.aliasIndexImage = (
			vkof::image_create({
				.width = (u32)width,
				.height = (u32)height,
				.depth = 1u,
				.format = vkof::ImageFormat::r32_float,
				.mipLevels = 1u,
				.optInitialData = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(aliasIndex.data()),
					sizeof(f32) * width * height
				),
			})
		),
		.radianceHandle = {},
		.pdfHandle = {},
		.aliasProbHandle = {},
		.aliasIndexHandle = {},
		.width = (u32)width,
		.height = (u32)height,
		.radianceMipCount = radianceMipCount,
	};
	vkof::image_generate_mipmaps(tables.radianceImage);
	tables.radianceHandle = (
		vkof::image_sampler_handle({
			.image = tables.radianceImage,
			.sampler = tables.radianceSampler,
		})
	);
	tables.pdfHandle = (
		vkof::image_sampler_handle({
			.image = tables.pdfImage,
			.sampler = tables.tableSampler,
		})
	);
	tables.aliasProbHandle = (
		vkof::image_sampler_handle({
			.image = tables.aliasProbImage,
			.sampler = tables.tableSampler,
		})
	);
	tables.aliasIndexHandle = (
		vkof::image_sampler_handle({
			.image = tables.aliasIndexImage,
			.sampler = tables.tableSampler,
		})
	);
	free(pixels);
	return tables;
}

void ponder::environment_tables_destroy(EnvironmentTables & tables) {
	vkof::image_destroy(tables.radianceImage);
	vkof::image_destroy(tables.pdfImage);
	vkof::image_destroy(tables.aliasProbImage);
	vkof::image_destroy(tables.aliasIndexImage);
	vkof::sampler_destroy(tables.radianceSampler);
	vkof::sampler_destroy(tables.tableSampler);
	tables = {};
}
