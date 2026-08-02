#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

namespace
{
	struct FillPush {
		u64 dstVa;
		u32 value;
		u32 count;
	};
}

static constexpr u32 skLocalSize = 64u;

TEST_SUITE("[headless]") {

// one render node: fill clears, a dispatch writes the indirect arguments, and
// an indirect dispatch reads them; every step is ordered by
// cmd_barrier_compute alone
TEST_CASE("compute: indirect dispatch reads arguments written in the same node") {
	u32 const elementCount = 1024u;
	u32 const indirectGroups = 5u;
	u32 const indirectValue = 7u;

	vkof::Pipeline const pipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = TEST_SHADER_DIR "slang/entry/fill_u32.slang",
			.includePaths = test::slang_include_paths(),
		})
	);
	REQUIRE(pipeline.id != 0u);
	vkof::Buffer const output = test::make_buffer_u32(elementCount);
	vkof::Buffer const arguments = test::make_buffer_u32(3u);

	// -- poison the output, so the fill must run for the tail to read zero
	std::vector<u32> const poison(elementCount, 0xdeadbeefu);
	vkof::buffer_upload({
		.buffer = output,
		.byteOffset = 0u,
		.data = (
			srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(poison.data()),
				elementCount * sizeof(u32)
			)
		),
	});
	vkof::validation_reset();

	vkof::RenderNode const node = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			// -- clear the output, and set the y and z group counts to 1
			vkof::cmd_fill_buffer({
				.cmd = cmd,
				.buffer = output,
				.byteOffset = 0u,
				.byteCount = elementCount * sizeof(u32),
				.value = 0u,
			});
			vkof::cmd_fill_buffer({
				.cmd = cmd,
				.buffer = arguments,
				.byteOffset = sizeof(u32),
				.byteCount = 2u * sizeof(u32),
				.value = 1u,
			});
			// -- a shader writes the x group count
			vkof::cmd_dispatch_pushconst(
				vkof::CmdDispatchPushconst<FillPush> {
					.cmd = cmd,
					.pipeline = pipeline,
					.push = (
						FillPush {
							.dstVa = vkof::buffer_virtual_address(arguments),
							.value = indirectGroups,
							.count = 1u,
						}
					),
					.threadgroupSize = u32v3 { skLocalSize, 1u, 1u },
					.invocationCount = u32v3 { 1u, 1u, 1u },
				}
			);
			vkof::cmd_barrier_compute(cmd);
			// -- writes indirectGroups * skLocalSize elements
			vkof::cmd_dispatch_indirect_pushconst(
				vkof::CmdDispatchIndirectPushconst<FillPush> {
					.cmd = cmd,
					.pipeline = pipeline,
					.push = (
						FillPush {
							.dstVa = vkof::buffer_virtual_address(output),
							.value = indirectValue,
							.count = elementCount,
						}
					),
					.argumentBuffer = arguments,
					.argumentByteOffset = 0u,
				}
			);
		},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);

	std::vector<u32> const result = (
		test::readback<u32>(output, 0u, elementCount)
	);
	u32 const writtenCount = indirectGroups * skLocalSize;
	u32 mismatches = 0u;
	for (u32 i = 0u; i < elementCount; ++i) {
		u32 const expected = i < writtenCount ? indirectValue : 0u;
		mismatches += result[i] != expected ? 1u : 0u;
	}
	CHECK(mismatches == 0u);
	CHECK(result[writtenCount - 1u] == indirectValue);
	CHECK(result[writtenCount] == 0u);
	CHECK(vkof::validation_message_count() == 0u);

	vkof::buffer_destroy(arguments);
	vkof::buffer_destroy(output);
	vkof::pipeline_destroy(pipeline);
}

}
