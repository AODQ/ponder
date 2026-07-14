#include <ponder/energy-tables.hpp>

namespace ponder {

// -----------------------------------------------------------------------------
// -- kulla-conty GGX directional albedo, 16x16 (roughness x mu)
// -----------------------------------------------------------------------------

static f32 const skKullaContyEnergyData[256] = {
	1.000000f, 0.988071f, 0.905022f, 0.896458f, 0.922060f, 0.940172f,
	0.951033f, 0.953774f, 0.952107f, 0.947958f, 0.942332f, 0.934545f,
	0.925498f, 0.914913f, 0.903202f, 0.890433f,
	1.000000f, 0.998627f, 0.977963f, 0.924440f, 0.889060f, 0.886391f,
	0.891276f, 0.894988f, 0.894870f, 0.886240f, 0.876163f, 0.858471f,
	0.840526f, 0.817747f, 0.794919f, 0.769773f,
	1.000000f, 0.999548f, 0.992477f, 0.961348f, 0.915559f, 0.883532f,
	0.870649f, 0.864415f, 0.858654f, 0.846603f, 0.828853f, 0.808554f,
	0.783298f, 0.752251f, 0.720449f, 0.687631f,
	1.000000f, 0.999813f, 0.996053f, 0.978904f, 0.942906f, 0.901699f,
	0.871593f, 0.852289f, 0.836613f, 0.816990f, 0.796714f, 0.768985f,
	0.735167f, 0.701014f, 0.663193f, 0.624795f,
	1.000000f, 0.999856f, 0.997469f, 0.987360f, 0.960689f, 0.920756f,
	0.881842f, 0.849685f, 0.824418f, 0.800921f, 0.770320f, 0.738004f,
	0.701490f, 0.660073f, 0.619153f, 0.572905f,
	1.000000f, 0.999878f, 0.998320f, 0.991305f, 0.972025f, 0.937171f,
	0.895881f, 0.857017f, 0.822157f, 0.787703f, 0.755183f, 0.715447f,
	0.670402f, 0.624560f, 0.579233f, 0.532599f,
	1.000000f, 0.999895f, 0.998738f, 0.993239f, 0.978505f, 0.949633f,
	0.908792f, 0.865747f, 0.825601f, 0.782747f, 0.741589f, 0.698455f,
	0.648912f, 0.597526f, 0.546716f, 0.495422f,
	1.000000f, 0.999962f, 0.999180f, 0.994909f, 0.982861f, 0.958860f,
	0.921529f, 0.876808f, 0.828128f, 0.781164f, 0.733090f, 0.682346f,
	0.628276f, 0.576641f, 0.520170f, 0.461832f,
	1.000000f, 0.999967f, 0.999143f, 0.996110f, 0.986222f, 0.965474f,
	0.931820f, 0.886617f, 0.837798f, 0.783723f, 0.729957f, 0.672340f,
	0.615165f, 0.555356f, 0.493722f, 0.438420f,
	1.000000f, 0.999950f, 0.999315f, 0.996472f, 0.988006f, 0.971015f,
	0.939302f, 0.896990f, 0.844288f, 0.786255f, 0.726462f, 0.663577f,
	0.601563f, 0.537290f, 0.471909f, 0.415300f,
	1.000000f, 0.999953f, 0.999497f, 0.997208f, 0.990300f, 0.974460f,
	0.947030f, 0.905052f, 0.852056f, 0.790519f, 0.726396f, 0.657019f,
	0.587188f, 0.523369f, 0.454441f, 0.391267f,
	1.000000f, 0.999994f, 0.999558f, 0.997375f, 0.991503f, 0.977419f,
	0.951576f, 0.912996f, 0.860437f, 0.796573f, 0.726529f, 0.655100f,
	0.581643f, 0.509683f, 0.437398f, 0.372621f,
	1.000000f, 0.999956f, 0.999568f, 0.997899f, 0.992178f, 0.980203f,
	0.957207f, 0.919148f, 0.867690f, 0.802882f, 0.730433f, 0.650576f,
	0.571347f, 0.495317f, 0.423144f, 0.354806f,
	1.000000f, 0.999978f, 0.999652f, 0.997975f, 0.992989f, 0.982050f,
	0.960359f, 0.924918f, 0.875282f, 0.807544f, 0.732619f, 0.649359f,
	0.564623f, 0.483091f, 0.408167f, 0.339617f,
	1.000000f, 0.999969f, 0.999708f, 0.998276f, 0.993703f, 0.983939f,
	0.964226f, 0.930900f, 0.879820f, 0.816623f, 0.736827f, 0.649661f,
	0.560940f, 0.477082f, 0.397979f, 0.324305f,
	1.000000f, 0.999960f, 0.999734f, 0.998373f, 0.994381f, 0.985219f,
	0.966582f, 0.935156f, 0.886950f, 0.821114f, 0.742601f, 0.650476f,
	0.559640f, 0.466482f, 0.385075f, 0.313566f,
};

// -----------------------------------------------------------------------------
// -- kulla-conty hemispherical albedo average, 16x1 (roughness)
// -----------------------------------------------------------------------------

static f32 const skKullaContyEnergyAvgData[16] = {
	1.000000f, 0.999880f, 0.998510f, 0.994123f, 0.984244f, 0.967015f,
	0.940452f, 0.904163f, 0.858230f, 0.802818f, 0.741311f, 0.673700f,
	0.604681f, 0.536525f, 0.470419f, 0.408946f,
};

EnergyTables energy_tables_create() {
	EnergyTables tables = {};
	tables.sampler = vkof::sampler_create({
		.magFilter = vkof::SamplerFilter::nearest,
		.minFilter = vkof::SamplerFilter::nearest,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
		.mipmapMode = vkof::SamplerMipmapMode::nearest,
	});

	tables.kullaContyEnergyImage = vkof::image_create({
		.width = 16u,
		.height = 16u,
		.depth = 1u,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1u,
		.optInitialData = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(skKullaContyEnergyData),
			sizeof(skKullaContyEnergyData)
		),
	});
	tables.kullaContyEnergyHandle = vkof::image_sampler_handle({
		.image = tables.kullaContyEnergyImage,
		.sampler = tables.sampler,
	});

	tables.kullaContyEnergyAvgImage = vkof::image_create({
		.width = 16u,
		.height = 1u,
		.depth = 1u,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1u,
		.optInitialData = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(skKullaContyEnergyAvgData),
			sizeof(skKullaContyEnergyAvgData)
		),
	});
	tables.kullaContyEnergyAvgHandle = vkof::image_sampler_handle({
		.image = tables.kullaContyEnergyAvgImage,
		.sampler = tables.sampler,
	});

	return tables;
}

void energy_tables_destroy(EnergyTables & tables) {
	vkof::image_destroy(tables.kullaContyEnergyImage);
	vkof::image_destroy(tables.kullaContyEnergyAvgImage);
	vkof::sampler_destroy(tables.sampler);
	tables = {};
}

} // namespace ponder
