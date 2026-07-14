#pragma once

#include <vkof/vkof.hpp>

#include <filesystem>

namespace ponder {

// spatiotemporal blue noise scalar textures, used to seed gpu randomness
struct Bluenoise {
	static constexpr u32 kCount = 64u;

	vkof::Sampler commonSampler {};
	vkof::Image images[kCount] {};
	u32 handles[kCount] {};
	vkof::Buffer handleBuffer {};
};

// textureDir must contain the stbn_scalar_2Dx1Dx1D_128x128x64x1_<n>.png set
[[nodiscard]] Bluenoise bluenoise_create(
	std::filesystem::path const & textureDir
);
void bluenoise_destroy(Bluenoise & bn);

} // namespace ponder
