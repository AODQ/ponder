#include <ponder/environment-tables.hpp>

#include <tinyexr.h>

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

// computes per-row conditional 1D CDF, normalized
void compute_marginal_cdf(
	std::vector<float> & conditionalCdf,
	std::vector<float> & rowIntegral,
	std::vector<float> const & importance,
	int32_t const width,
	int32_t const height
) {
	// calculate normalized running sum
	conditionalCdf.resize((width+1) * height);
	rowIntegral.resize(height);
	for (int32_t y = 0; y < height; ++y) {
		f32 * const cdf = &conditionalCdf[y * (width + 1)];
		cdf[0] = 0.0f;
		for (int32_t x = 0; x < width; ++x) {
			cdf[x + 1] = cdf[x] + importance[y * width + x];
		}
		f32 const rowSum = cdf[width];
		rowIntegral[y] = rowSum;
		if (rowSum > 0.0f) {
			for (i32 x = 1; x <= width; ++x) {
				cdf[x] /= rowSum;
			}
		} else {
			// purely black row, no importance, make a uniform distribution
			for (i32 x = 1; x <= width; ++x) {
				cdf[x] = float(x) / float(width);
			}
		}
	}
}

// computed marginal 1D CDF, normalized
void compute_marginal_cdf(
	std::vector<float> & marginalCdf,
	std::vector<float> const & rowIntegral,
	f32 & totalSum,
	int32_t const height
) {
	marginalCdf.resize(height + 1);
	marginalCdf[0] = 0.0f;
	for (int32_t y = 0; y < height; ++y) {
		marginalCdf[y + 1] = marginalCdf[y] + rowIntegral[y];
	}
	totalSum = marginalCdf[height];
	if (totalSum > 0.0f) {
		for (i32 y = 1; y <= height; ++y) {
			marginalCdf[y] /= marginalCdf[height];
		}
	} else {
		// purely black image, no importance, make a uniform distribution
		for (i32 y = 1; y <= height; ++y) {
			marginalCdf[y] = float(y) / float(height);
		}
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

	// -- conditional cdf
	std::vector<float> conditionalCdf;
	std::vector<float> rowIntegral;
	compute_marginal_cdf(
		conditionalCdf,
		rowIntegral,
		importance,
		width,
		height
	);

	// -- marginal cdf
	std::vector<float> marginalCdf;
	f32 totalSum;
	compute_marginal_cdf(
		marginalCdf,
		rowIntegral,
		totalSum,
		height
	);

	// -- pdf
	std::vector<float> pdf;
	compute_pdf(
		pdf,
		importance,
		totalSum,
		width,
		height
	);

	// -- upload to GPU
	EnvironmentTables tables {
		.radianceSampler = vkof::sampler_create({
			.magFilter = vkof::SamplerFilter::linear,
			.minFilter = vkof::SamplerFilter::linear,
			.addressModeU = vkof::SamplerAddressMode::repeat,
			.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
			.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
			.mipmapMode = vkof::SamplerMipmapMode::nearest,
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
				.mipLevels = 1u,
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
		.conditionalCdfImage = (
			vkof::image_create({
				.width = (u32)(width + 1),
				.height = (u32)height,
				.depth = 1u,
				.format = vkof::ImageFormat::r32_float,
				.mipLevels = 1u,
				.optInitialData = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(conditionalCdf.data()),
					sizeof(f32) * (width + 1) * height
				),
			})
		),
		.marginalCdfImage = (
			vkof::image_create({
				.width = (u32)(height + 1),
				.height = 1u,
				.depth = 1u,
				.format = vkof::ImageFormat::r32_float,
				.mipLevels = 1u,
				.optInitialData = srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(marginalCdf.data()),
					sizeof(f32) * (height + 1)
				),
			})
		),
		.radianceHandle = {},
		.pdfHandle = {},
		.conditionalCdfHandle = {},
		.marginalCdfHandle = {},
		.width = (u32)width,
		.height = (u32)height,
	};
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
	tables.conditionalCdfHandle = (
		vkof::image_sampler_handle({
			.image = tables.conditionalCdfImage,
			.sampler = tables.tableSampler,
		})
	);
	tables.marginalCdfHandle = (
		vkof::image_sampler_handle({
			.image = tables.marginalCdfImage,
			.sampler = tables.tableSampler,
		})
	);
	free(pixels);
	return tables;
}

void ponder::environment_tables_destroy(EnvironmentTables & tables) {
	vkof::image_destroy(tables.radianceImage);
	vkof::image_destroy(tables.pdfImage);
	vkof::image_destroy(tables.conditionalCdfImage);
	vkof::image_destroy(tables.marginalCdfImage);
	vkof::sampler_destroy(tables.radianceSampler);
	vkof::sampler_destroy(tables.tableSampler);
	tables = {};
}
