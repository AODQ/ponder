#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <vector>

// ---------------------------------------------------------------------------
// diagnostic suite (not a milestone test): investigates a reproducible
// anomaly found while building the mixture-lobe-selection milestone
// (test-mixture-lobe-selection.cpp) -- a chi-square check that passes when
// run alone but fails when a *later-registered, heavier* test case is also
// selected via --test-case, even though that later case hasn't executed
// yet when the earlier one's assertions run. that rules out a bug in the
// mixture pdf math itself (order can't retroactively change an already-
// read-back result if readback is genuinely synchronous) and points at
// test::dispatch/test::readback or vkof's compute dispatch path instead.
//
// this file tries to reproduce the same shape of anomaly using only
// fill_u32.comp/passthrough.comp -- trivial, exact-value shaders with zero
// relation to openpbr -- so a repro here would confirm the bug lives in
// the shared harness, not in anything under verification.
// ---------------------------------------------------------------------------

namespace {

struct FillPush { u64 dstVa; u32 value; u32 count; };
static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

// one pipeline create + dispatch + readback + destroy, exact-value checked.
// mirrors the shape of a single mixture_histogram.comp/mixture_expected.comp
// call in test-mixture-lobe-selection.cpp, but with a trivial shader and a
// bit-exact (not statistical) correctness check.
void light_fill_check(u32 const count, u32 const value, char const * const label) {
	auto pl = vkof::pipeline_compute_create(
		{ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" }
	);
	REQUIRE(pl.id != 0);
	auto buf = test::make_buffer_u32(count);
	FillPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.value = value,
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));
	auto result = test::readback<u32>(buf, 0, count);

	u32 wrongCount = 0;
	u32 firstWrongIndex = count;
	u32 firstWrongValue = 0;
	for (u32 i = 0; i < count; ++i) {
		if (result[i] != value) {
			wrongCount++;
			if (firstWrongIndex == count) {
				firstWrongIndex = i;
				firstWrongValue = result[i];
			}
		}
	}
	CAPTURE(label);
	CAPTURE(count);
	CAPTURE(value);
	CAPTURE(wrongCount);
	CAPTURE(firstWrongIndex);
	CAPTURE(firstWrongValue);
	CHECK(wrongCount == 0u);

	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

} // namespace

TEST_SUITE("[headless]") {

// registered first: this is the case whose pass/fail we're watching for a
// change depending on whether the heavier case below is also selected
TEST_CASE("DIAGNOSTIC: light dispatch+readback, exact value, 3200 elements") {
	// 3200 == BSDF_VERIFY_BIN_COUNT, matching the mixture chi-square
	// histogram/expected buffer size exactly
	light_fill_check(3200u, 0xABCD1234u, "light-alone");
}

// registered second, much heavier: creates/destroys many pipelines and
// dispatches repeatedly, mirroring check_mixture_chi_square's per-config
// pipeline churn (histogram + expected + selection-crosscheck, x3 ct
// values, x2 shaders each) using only the trivial fill shader
TEST_CASE("DIAGNOSTIC: heavy churn, many pipeline create/dispatch/destroy cycles") {
	for (u32 iter = 0; iter < 30; ++iter) {
		light_fill_check(3200u, 0x1000u + iter, "heavy-churn-iter");
	}
}

// same light check again, registered last -- if the anomaly is about
// *total* churn rather than specifically "did the heavy case run before
// me," this should fail even though it runs after the heavy case has
// definitely completed
TEST_CASE("DIAGNOSTIC: light dispatch+readback again, after heavy churn ran") {
	light_fill_check(3200u, 0xEF012345u, "light-after-heavy");
}

} // TEST_SUITE("[headless]")
