#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

// smoke test for vkof::validation_message_count/validation_message/
// validation_reset (lib/vkof/src/vkof.cpp): the debug messenger callback
// that already backs vkof::probe_message (exercised by
// test-mixture-lobe-selection.cpp) now also archives every non-debug-
// printf, non-info validation message it sees, so test code can assert
// "no validation errors happened" instead of relying on someone reading
// stderr. this only proves the wiring on the common (no-error) path --
// deliberately provoking a real validation error to prove the capture
// side wasn't attempted here (too likely to wedge the driver/test run).

struct FillPush { u64 dstVa; u32 value; u32 count; };
static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

TEST_SUITE("[headless]") {

TEST_CASE("vkof validation capture: reset clears the log") {
	// unrelated tests earlier in the run may have left messages behind;
	// reset must always bring the count to zero regardless of history
	vkof::validation_reset();
	CHECK(vkof::validation_message_count() == 0u);
}

TEST_CASE("vkof validation capture: a benign dispatch logs nothing") {
	vkof::validation_reset();

	constexpr u32 kCount = 1024;
	auto pl = vkof::pipeline_compute_create(
		{ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" }
	);
	REQUIRE(pl.id != 0);
	auto buf = test::make_buffer_u32(kCount);
	FillPush const push { .dstVa = vkof::buffer_virtual_address(buf), .value = 7u, .count = kCount };
	test::dispatch(pl, push, groups_for(kCount));
	auto result = test::readback<u32>(buf, 0, kCount);

	u32 const count = vkof::validation_message_count();
	if (count > 0u) {
		MESSAGE("unexpected validation message: ", vkof::validation_message(0u));
	}
	CHECK(count == 0u);
	CHECK(result[0] == 7u);

	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

} // TEST_SUITE("[headless]")
