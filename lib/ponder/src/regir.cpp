#include <ponder/regir.hpp>

#include <string>
#include <vector>

namespace {

static constexpr char const * skPonderShaderNewDir = PONDER_SHADER_NEW_DIR;

// has to be power of two for the regir fill shader to work
static constexpr u32 skRegirCellCount = 131'072u;

// mirrors the regir fill entry's stage push constants (at offset 128)
struct RegirFillPc {
	u32 pad0;
};
static_assert(sizeof(RegirFillPc) == 4u);

// mirrors the regir view entry's stage push constants (at offset 128)
struct RegirViewPc {
	u32 outputHandle;
	u32 mode;
	f32 exposure;
};
static_assert(sizeof(RegirViewPc) == 12u);

char const * const skIncludePaths[] = { skPonderShaderNewDir };
std::string const skRegirFillPath = (
	std::string(skPonderShaderNewDir) + "entry/regir-fill.slang"
);
std::string const skRegirViewPath = (
	std::string(skPonderShaderNewDir) + "entry/regir-view.slang"
);

vkof::PipelineComputeCreateInfo compute_create_info(std::string const & path) {
	return vkof::PipelineComputeCreateInfo {
		.pathCompute = path.c_str(),
		.defines = {},
		.includePaths = srat::slice { skIncludePaths, 1u },
	};
}

void guards_zero(vkof::Buffer const buffer, u32 const cellCount) {
	std::vector<u32> const zeros(cellCount, 0u);
	vkof::buffer_upload({
		.buffer = buffer,
		.byteOffset = 0u,
		.data = (
			srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(zeros.data()),
				(u32)(sizeof(u32) * zeros.size())
			)
		),
	});
}

} // namespace

ponder::Regir ponder::regir_create() {
	auto const guardBuffer = (
		vkof::buffer_create({
			.byteCount = sizeof(u32) * skRegirCellCount,
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);
	guards_zero(guardBuffer, skRegirCellCount);
	auto const cellBuffer = (
		vkof::buffer_create({
			.byteCount = sizeof(RegirCell) * skRegirCellCount,
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);
	auto const reservoirBuffer = (
		vkof::buffer_create({
			.byteCount = (
				(u64)sizeof(RegirReservoir)
				* skRegirCellCount
				* skWavefrontRegirReservoirCount
			),
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);
	return Regir {
		.fillPipeline = (
			vkof::pipeline_compute_create(compute_create_info(skRegirFillPath))
		),
		.viewPipeline = {},
		.guardBuffer = guardBuffer,
		.cellBuffer = cellBuffer,
		.reservoirBuffer = reservoirBuffer,
		.cellCount = skRegirCellCount,
	};
}

void ponder::regir_destroy(Regir & regir) {
	vkof::pipeline_destroy(regir.fillPipeline);
	if (regir.viewPipeline.id) {
		vkof::pipeline_destroy(regir.viewPipeline);
	}
	vkof::buffer_destroy(regir.guardBuffer);
	vkof::buffer_destroy(regir.cellBuffer);
	vkof::buffer_destroy(regir.reservoirBuffer);
	regir = {};
}

vkof::RenderNode ponder::regir_fill(Regir const & regir) {
	auto const node = (
		vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
			.debugName = "regir fill",
		})
	);
	vkof::Pipeline const pipeline = regir.fillPipeline;
	auto const pc = RegirFillPc { .pad0 = 0u };
	u32 const threadCount = regir.cellCount * skWavefrontRegirReservoirCount;
	vkof::render_node_callback({
		.node = node,
		.callback = [=](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
				.cmd = cmd,
				.pipeline = pipeline,
				.push = pc,
				.threadgroupSize = u32v3 { skWavefrontRegirReservoirCount, 1u, 1u },
				.invocationCount = u32v3 { threadCount, 1u, 1u },
			});
		},
	});
	return node;
}

void ponder::regir_clear(Regir const & regir) {
	vkof::device_wait_idle();
	guards_zero(regir.guardBuffer, regir.cellCount);
}

void ponder::regir_view_prepare(Regir & regir) {
	if (regir.viewPipeline.id) {
		return;
	}
	regir.viewPipeline = (
		vkof::pipeline_compute_create(compute_create_info(skRegirViewPath))
	);
}

vkof::RenderNode ponder::regir_view(
	Regir const & regir,
	RegirViewInfo const & info
) {
	auto const node = (
		vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
			.debugName = "regir view",
		})
	);
	vkof::render_node_add_image({
		.node = node,
		.image = info.outputImage,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::Pipeline const pipeline = regir.viewPipeline;
	auto const pc = RegirViewPc {
		.outputHandle = info.outputHandle,
		.mode = info.mode,
		.exposure = info.exposure,
	};
	u32 const width = info.width;
	u32 const height = info.height;
	vkof::render_node_callback({
		.node = node,
		.callback = [=](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
				.cmd = cmd,
				.pipeline = pipeline,
				.push = pc,
				.threadgroupSize = u32v3 { 16u, 16u, 1u },
				.invocationCount = u32v3 { width, height, 1u },
			});
		},
	});
	return node;
}
