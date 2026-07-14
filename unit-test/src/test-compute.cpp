
#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

// Push structs for our test shaders.
// These are pushed at byte-offset 128 inside the universal push-constant range.
struct FillPush {
	u64 dstVa;
	u32 value;
	u32 count;
};

struct PassthroughPush {
	u64 srcVa;
	u64 dstVa;
	u32 count;
	u32 _pad = 0;
};

static constexpr u32 kLocalSize = 64; // must match shader local_size_x

static u32 groups_for(u32 count) {
	return (count + kLocalSize - 1) / kLocalSize;
}

TEST_SUITE("[headless]") {

TEST_CASE("compute: fill_u32 basic") {
	constexpr u32 kCount = 1024;
	constexpr u32 kValue = 42;

	auto pl  = (
		vkof::pipeline_compute_create(
			{ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" }
		)
	);
	auto buf = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	FillPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.value  = kValue,
		.count  = kCount,
	};
	test::dispatch(pl, push, groups_for(kCount));
	auto result = test::readback<u32>(buf, 0, kCount);

	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == kValue);
	}
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("compute: fill_u32 zero value") {
	constexpr u32 kCount = 256;
	constexpr u32 kValue = 0;

	auto pl  = (
		vkof::pipeline_compute_create(
			{ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" }
		)
	);
	auto buf = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	// Pre-fill with a sentinel so we can verify the zero-fill actually ran
	std::vector<u32> sentinel(kCount, 0xDEADBEEFu);
	vkof::buffer_upload({
		.buffer = buf,
		.byteOffset = 0,
		.data   = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(sentinel.data()),
			kCount * sizeof(u32)
		),
	});

	FillPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.value  = kValue,
		.count  = kCount,
	};
	test::dispatch(pl, push, groups_for(kCount));
	auto result = test::readback<u32>(buf, 0, kCount);

	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == kValue);
	}
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("compute: fill_u32 max value") {
	constexpr u32 kCount = 256;
	constexpr u32 kValue = 0xFFFFFFFFu;

	auto pl  = (
		vkof::pipeline_compute_create(
			{ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" }
		)
	);
	auto buf = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	FillPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.value  = kValue,
		.count  = kCount,
	};
	test::dispatch(pl, push, groups_for(kCount));
	auto result = test::readback<u32>(buf, 0, kCount);

	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == kValue);
	}
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("compute: fill_u32 count=1") {
	auto pl  = (
		vkof::pipeline_compute_create(
			{ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" }
		)
	);
	auto buf = test::make_buffer_u32(64); // allocate at least 4 bytes
	REQUIRE(pl.id != 0);

	FillPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.value  = 999u,
		.count  = 1,
	};
	test::dispatch(pl, push, 1);
	auto result = test::readback<u32>(buf, 0, 1);

	CHECK(result[0] == 999u);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("compute: fill_u32 one workgroup exactly") {
	constexpr u32 kCount = kLocalSize;
	constexpr u32 kValue = 0xCAFEBABEu;

	auto pl  = (
		vkof::pipeline_compute_create(
			{ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" }
		)
	);
	auto buf = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	FillPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.value  = kValue,
		.count  = kCount,
	};
	test::dispatch(pl, push, 1); // exactly 1 group
	auto result = test::readback<u32>(buf, 0, kCount);

	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == kValue);
	}
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("compute: passthrough copy") {
	constexpr u32 kCount = 1024;

	auto pl  = (
		vkof::pipeline_compute_create(
			{ .pathCompute = TEST_SHADER_DIR "passthrough.comp" }
		)
	);
	auto src = test::make_buffer_u32(kCount);
	auto dst = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	// Upload a known pattern to src
	std::vector<u32> pattern(kCount);
	for (u32 i = 0; i < kCount; ++i) { pattern[i] = i * 13 + 7; }
	vkof::buffer_upload({
		.buffer	 = src,
		.byteOffset = 0,
		.data	   = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(pattern.data()),
			kCount * sizeof(u32)
		),
	});

	PassthroughPush const push {
		.srcVa = vkof::buffer_virtual_address(src),
		.dstVa = vkof::buffer_virtual_address(dst),
		.count  = kCount,
	};
	test::dispatch(pl, push, groups_for(kCount));
	auto result = test::readback<u32>(dst, 0, kCount);

	for (u32 i = 0; i < kCount; ++i) {
		CAPTURE(i);
		CHECK(result[i] == pattern[i]);
	}
	vkof::buffer_destroy(src);
	vkof::buffer_destroy(dst);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("compute: passthrough large copy") {
	constexpr u32 kCount = 4096;

	auto pl  = (
		vkof::pipeline_compute_create(
			{ .pathCompute = TEST_SHADER_DIR "passthrough.comp" }
		)
	);
	auto src = test::make_buffer_u32(kCount);
	auto dst = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	std::vector<u32> pattern(kCount);
	for (u32 i = 0; i < kCount; ++i) { pattern[i] = ~i; }
	vkof::buffer_upload({
		.buffer	 = src,
		.byteOffset = 0,
		.data	   = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(pattern.data()),
			kCount * sizeof(u32)
		),
	});

	PassthroughPush const push {
		.srcVa = vkof::buffer_virtual_address(src),
		.dstVa = vkof::buffer_virtual_address(dst),
		.count  = kCount,
	};
	test::dispatch(pl, push, groups_for(kCount));
	auto result = test::readback<u32>(dst, 0, kCount);

	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == ~i);
	}
	vkof::buffer_destroy(src);
	vkof::buffer_destroy(dst);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("compute: sequential dispatches") {
	constexpr u32 kCount = 256;

	auto pl  = (
		vkof::pipeline_compute_create(
			{ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" }
		)
	);
	auto buf = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	// First dispatch: fill with 111
	{
		FillPush const push {
			.dstVa = vkof::buffer_virtual_address(buf),
			.value  = 111u,
			.count  = kCount,
		};
		test::dispatch(pl, push, groups_for(kCount));
	}
	// Second dispatch: fill with 222
	{
		FillPush const push {
			.dstVa = vkof::buffer_virtual_address(buf),
			.value  = 222u,
			.count  = kCount,
		};
		test::dispatch(pl, push, groups_for(kCount));
	}
	auto result = test::readback<u32>(buf, 0, kCount);

	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == 222u);
	}
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

} // TEST_SUITE("[headless]")
