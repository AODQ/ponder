#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"
#include <cstring>
#include <vector>

TEST_SUITE("[headless]") {

TEST_CASE("buffer: create/destroy DeviceOnly") {
	auto const buf = vkof::buffer_create({
		.byteCount = 256,
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	CHECK(buf.id != 0);
	vkof::buffer_destroy(buf);
}

TEST_CASE("buffer: create/destroy HostWritable") {
	auto const buf = vkof::buffer_create({
		.byteCount = 256,
		.memory = vkof::BufferMemory::HostWritable,
	});
	CHECK(buf.id != 0);
	vkof::buffer_destroy(buf);
}

TEST_CASE("buffer: DeviceOnly Va is non-zero") {
	auto const buf = vkof::buffer_create({
		.byteCount = 64,
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	CHECK(vkof::buffer_virtual_address(buf) != 0u);
	vkof::buffer_destroy(buf);
}

TEST_CASE("buffer: HostWritable Va is non-zero") {
	auto const buf = vkof::buffer_create({
		.byteCount = 64,
		.memory = vkof::BufferMemory::HostWritable,
	});
	CHECK(vkof::buffer_virtual_address(buf) != 0u);
	vkof::buffer_destroy(buf);
}

TEST_CASE("buffer: HostWritable host_address is valid") {
	constexpr u64 kSize = 128;
	auto const buf = vkof::buffer_create({
		.byteCount = kSize,
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const slice = vkof::buffer_host_address(buf);
	CHECK(slice.ptr() != nullptr);
	CHECK(slice.size() == kSize);
	vkof::buffer_destroy(buf);
}

TEST_CASE("buffer: DeviceOnly host_address is empty") {
	auto const buf = vkof::buffer_create({
		.byteCount = 128,
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto const slice = vkof::buffer_host_address(buf);
	CHECK(slice.ptr() == nullptr);
	vkof::buffer_destroy(buf);
}

TEST_CASE("buffer: upload/download round-trip") {
	constexpr u32 kCount = 64;
	constexpr u64 kBytes = kCount * sizeof(u32);

	// Source data
	std::vector<u32> src(kCount);
	for (u32 i = 0; i < kCount; ++i) { src[i] = i * 7 + 13; }

	auto const buf = vkof::buffer_create({
		.byteCount = kBytes,
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	vkof::buffer_upload({
		.buffer	 = buf,
		.byteOffset = 0,
		.data	   = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(src.data()), kBytes
		),
	});

	auto const result = test::readback<u32>(buf, 0, kCount);

	for (u32 i = 0; i < kCount; ++i) {
		CAPTURE(i);
		CHECK(result[i] == src[i]);
	}
	vkof::buffer_destroy(buf);
}

// 8. Two distinct buffers have distinct Va
TEST_CASE("buffer: distinct Vas") {
	auto const b0 = (
		vkof::buffer_create(
			{ .byteCount = 64, .memory = vkof::BufferMemory::DeviceOnly }
		)
	);
	auto const b1 = (
		vkof::buffer_create(
			{ .byteCount = 64, .memory = vkof::BufferMemory::DeviceOnly }
		)
	);
	CHECK(vkof::buffer_virtual_address(b0) != vkof::buffer_virtual_address(b1));
	vkof::buffer_destroy(b0);
	vkof::buffer_destroy(b1);
}

// 9. After destroy, buffer_virtual_address returns 0
TEST_CASE("buffer: Va is 0 after destroy") {
	auto const buf = vkof::buffer_create({
		.byteCount = 64,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_destroy(buf);
	CHECK(vkof::buffer_virtual_address(buf) == 0u);
}

// 10. After destroy, buffer_host_address returns an empty slice
TEST_CASE("buffer: host_address empty after destroy") {
	auto const buf = vkof::buffer_create({
		.byteCount = 64,
		.memory	= vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_destroy(buf);
	auto const slice = vkof::buffer_host_address(buf);
	CHECK(slice.ptr() == nullptr);
}

// 11. Minimum-size (4-byte) buffer works
TEST_CASE("buffer: 4-byte buffer") {
	auto const buf = vkof::buffer_create({
		.byteCount = 4,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	CHECK(buf.id != 0);
	CHECK(vkof::buffer_virtual_address(buf) != 0u);
	vkof::buffer_destroy(buf);
}

// 12. Large (1 MB) buffer works
TEST_CASE("buffer: 1MB DeviceOnly buffer") {
	constexpr u64 kOneMB = 1024u * 1024u;
	auto const buf = vkof::buffer_create({
		.byteCount = kOneMB,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	CHECK(buf.id != 0);
	CHECK(vkof::buffer_virtual_address(buf) != 0u);
	vkof::buffer_destroy(buf);
}

// 13. buffer_upload with non-zero byteOffset leaves preceding bytes untouched
TEST_CASE("buffer: upload with non-zero byte offset") {
	constexpr u32 kTotal = 32u;
	constexpr u32 kSkip = 4u;
	constexpr u64 kSkipBytes = kSkip * sizeof(u32);
	constexpr u32 kUpload = kTotal - kSkip;

	auto const buf = vkof::buffer_create({
		.byteCount = kTotal * sizeof(u32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	std::vector<u32> sentinel(kTotal, 0xDEADBEEFu);
	vkof::buffer_upload({
		.buffer = buf,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(sentinel.data()),
			kTotal * sizeof(u32)
		),
	});

	std::vector<u32> patch(kUpload);
	for (u32 i = 0u; i < kUpload; ++i) { patch[i] = i + 1u; }
	vkof::buffer_upload({
		.buffer = buf,
		.byteOffset = kSkipBytes,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(patch.data()),
			kUpload * sizeof(u32)
		),
	});

	auto const result = test::readback<u32>(buf, 0u, kTotal);
	for (u32 i = 0u; i < kSkip; ++i) {
		CAPTURE(i);
		CHECK(result[i] == 0xDEADBEEFu);
	}
	for (u32 i = 0u; i < kUpload; ++i) {
		CAPTURE(i);
		CHECK(result[kSkip + i] == patch[i]);
	}

	vkof::buffer_destroy(buf);
}

TEST_CASE("buffer: rapid create/destroy cycle") {
	for (u32 i = 0u; i < 256u; i++) {
		vkof::Buffer const buf = vkof::buffer_create({
			.byteCount = 64u,
			.memory = vkof::BufferMemory::DeviceOnly,
		});
		CHECK(buf.id != 0u);
		vkof::buffer_destroy(buf);
	}
}

} // TEST_SUITE("[headless]")
