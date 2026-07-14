#include <ponder/bluenoise.hpp>

#include <stb_image.h>

#include <cstdio>
#include <string>

namespace ponder {

Bluenoise bluenoise_create(std::filesystem::path const & textureDir)
{
	Bluenoise bn = {};
	bn.commonSampler = vkof::sampler_create({
		.magFilter = vkof::SamplerFilter::nearest,
		.minFilter = vkof::SamplerFilter::nearest,
		.addressModeU = vkof::SamplerAddressMode::repeat,
		.addressModeV = vkof::SamplerAddressMode::repeat,
		.addressModeW = vkof::SamplerAddressMode::repeat,
		.mipmapMode = vkof::SamplerMipmapMode::nearest,
	});
	for (u32 i = 0u; i < Bluenoise::kCount; ++i) {
		std::string const path = (
			textureDir
			/ ("stbn_scalar_2Dx1Dx1D_128x128x64x1_" + std::to_string(i) + ".png")
		).string();
		int w = 0, h = 0, channels = 0;
		stbi_uc * const pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
		if (!pixels) {
			printf("[warn] bluenoise: failed to load %s\n", path.c_str());
			continue;
		}
		bn.images[i] = vkof::image_create({
			.width = (u32)w,
			.height = (u32)h,
			.depth = 1u,
			.format = vkof::ImageFormat::r8g8b8a8_unorm,
			.mipLevels = 1u,
			.optInitialData = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(pixels),
				(u32)(w * h * 4)
			),
		});
		stbi_image_free(pixels);
		bn.handles[i] = vkof::image_sampler_handle({
			.image = bn.images[i],
			.sampler = bn.commonSampler,
		});
	}
	bn.handleBuffer = vkof::buffer_create({
		.byteCount = sizeof(u32) * Bluenoise::kCount,
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = bn.handleBuffer,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(bn.handles),
			sizeof(u32) * Bluenoise::kCount
		),
	});
	return bn;
}

void bluenoise_destroy(Bluenoise & bn)
{
	vkof::buffer_destroy(bn.handleBuffer);
	for (u32 i = 0u; i < Bluenoise::kCount; ++i) {
		if (bn.images[i].id) { vkof::image_destroy(bn.images[i]); }
	}
	vkof::sampler_destroy(bn.commonSampler);
	bn = {};
}

} // namespace ponder
