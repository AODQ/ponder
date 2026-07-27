#include <ponder/environment-ibl.hpp>

#include "env-prefilter-pc.h"

#include <algorithm>

ponder::EnvironmentIbl ponder::environment_ibl_create() {
	EnvironmentIbl ibl;
	ibl.sampler = vkof::sampler_create({
		.magFilter = vkof::SamplerFilter::linear,
		.minFilter = vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::repeat,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
		.mipmapMode = vkof::SamplerMipmapMode::linear,
	});
	ibl.specularImage = vkof::image_create({
		.width = EnvironmentIbl::skSpecularBaseWidth,
		.height = EnvironmentIbl::skSpecularBaseHeight,
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels = EnvironmentIbl::skSpecularMipCount,
		.optInitialData = {},
	});
	ibl.irradianceImage = vkof::image_create({
		.width = EnvironmentIbl::skIrradianceWidth,
		.height = EnvironmentIbl::skIrradianceHeight,
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels = 1u,
		.optInitialData = {},
	});
	ibl.specularHandle = vkof::image_sampler_handle({
		.image = ibl.specularImage,
		.sampler = ibl.sampler,
	});
	ibl.irradianceHandle = vkof::image_sampler_handle({
		.image = ibl.irradianceImage,
		.sampler = ibl.sampler,
	});

	char const * const includePaths[] = { PONDER_SHADER_DIR, MOR_INCLUDE_DIR };
	ibl.specularPipeline = vkof::pipeline_compute_create({
		.pathCompute = PONDER_SHADER_DIR "env-prefilter-specular.comp",
		.includePaths = srat::slice { includePaths, 2u },
	});
	ibl.irradiancePipeline = vkof::pipeline_compute_create({
		.pathCompute = PONDER_SHADER_DIR "env-prefilter-irradiance.comp",
		.includePaths = srat::slice { includePaths, 2u },
	});

	return ibl;
}

void ponder::environment_ibl_destroy(EnvironmentIbl & ibl) {
	vkof::pipeline_destroy(ibl.specularPipeline);
	vkof::pipeline_destroy(ibl.irradiancePipeline);
	vkof::image_destroy(ibl.specularImage);
	vkof::image_destroy(ibl.irradianceImage);
	vkof::sampler_destroy(ibl.sampler);
	ibl = {};
}

void ponder::environment_ibl_bake(
	EnvironmentIbl const & ibl,
	EnvironmentTables const & env,
	std::vector<vkof::RenderNode> & nodes
) {
	for (u32 mip = 0u; mip < EnvironmentIbl::skSpecularMipCount; ++mip) {
		u32 const mipW = (
			std::max(1u, EnvironmentIbl::skSpecularBaseWidth >> mip)
		);
		u32 const mipH = (
			std::max(1u, EnvironmentIbl::skSpecularBaseHeight >> mip)
		);
		f32 const roughness = (
			(f32)mip / (f32)(EnvironmentIbl::skSpecularMipCount - 1u)
		);
		vkof::RenderNode const node = vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
		});
		vkof::render_node_add_persistent_image({
			.node = node,
			.image = ibl.specularImage,
			.access = vkof::RenderNodeAccess::write,
		});
		u32 const dstHandle = vkof::image_storage_handle({
			.image = ibl.specularImage,
			.mipLevel = mip,
		});
		u32 const srcHandle = env.radianceHandle;
		u32 const srcMipCount = env.radianceMipCount;
		vkof::Pipeline const pipeline = ibl.specularPipeline;
		// rougher mips need more samples to average out a bright, small
		// source without fireflying; fewer texels there offsets the cost
		u32 const sampleCount = 256u * (mip + 1u);
		vkof::render_node_callback({
			.node = node,
			.callback = [=](vkof::CommandBuffer const & cmd) {
				GpuEnvPrefilterPc const prefilterPc = {
					.srcRadianceHandle = srcHandle,
					.srcRadianceMipCount = srcMipCount,
					.dstStorageHandle = dstHandle,
					.dstWidth = mipW,
					.dstHeight = mipH,
					.roughness = roughness,
					.sampleCount = sampleCount,
				};
				vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
					.cmd = cmd,
					.pipeline = pipeline,
					.push = prefilterPc,
					.threadgroupSize = u32v3 { 8u, 8u, 1u },
					.invocationCount = u32v3 { mipW, mipH, 1u },
				});
			},
		});
		nodes.emplace_back(node);
	}

	vkof::RenderNode const irradianceNode = vkof::render_node_create({
		.queue = vkof::CommandQueue::graphics,
	});
	vkof::render_node_add_persistent_image({
		.node = irradianceNode,
		.image = ibl.irradianceImage,
		.access = vkof::RenderNodeAccess::write,
	});
	u32 const irradianceDstHandle = vkof::image_storage_handle({
		.image = ibl.irradianceImage,
		.mipLevel = 0u,
	});
	u32 const irradianceSrcHandle = env.radianceHandle;
	u32 const irradianceSrcMipCount = env.radianceMipCount;
	vkof::Pipeline const irradiancePipeline = ibl.irradiancePipeline;
	vkof::render_node_callback({
		.node = irradianceNode,
		.callback = [=](vkof::CommandBuffer const & cmd) {
			GpuEnvPrefilterPc const prefilterPc = {
				.srcRadianceHandle = irradianceSrcHandle,
				.srcRadianceMipCount = irradianceSrcMipCount,
				.dstStorageHandle = irradianceDstHandle,
				.dstWidth = EnvironmentIbl::skIrradianceWidth,
				.dstHeight = EnvironmentIbl::skIrradianceHeight,
				.roughness = 0.0f,
				.sampleCount = 256u,
			};
			vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
				.cmd = cmd,
				.pipeline = irradiancePipeline,
				.push = prefilterPc,
				.threadgroupSize = u32v3 { 8u, 8u, 1u },
				.invocationCount = (
					u32v3 {
						EnvironmentIbl::skIrradianceWidth,
						EnvironmentIbl::skIrradianceHeight,
						1u,
					}
				),
			});
		},
	});
	nodes.emplace_back(irradianceNode);
}
