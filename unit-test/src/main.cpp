// unit-test/main.cpp
// doctest entry point: parse --windowed, init vkof, run tests, shutdown.

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <vkof/vkof.hpp>
#include <cstring>
#include <vector>

extern "C" const char * __asan_default_options() {
	return "protect_shadow_gap=0:fast_unwind_on_malloc=0";
}

// diagnostic: a chi-square check in test-mixture-lobe-selection.cpp fails
// deterministically depending on which *other* test cases are also
// selected via --test-case, even when it finishes and reads back before
// those other cases run -- ruling out a bug in the ported glsl itself
// (same formula, same inputs, can't depend on unrelated doctest
// filtering) and pointing at state carried across dispatches on the gpu
// side. this idles the device between every test case to see whether the
// anomaly is sensitive to in-flight work overlapping across test-case
// boundaries.
struct DeviceIdleListener : doctest::IReporter {
	explicit DeviceIdleListener(const doctest::ContextOptions &) {}
	void report_query(const doctest::QueryData &) override {}
	void test_run_start() override {}
	void test_run_end(const doctest::TestRunStats &) override {}
	void test_case_start(const doctest::TestCaseData &) override {}
	void test_case_reenter(const doctest::TestCaseData &) override {}
	void test_case_end(const doctest::CurrentTestCaseStats &) override {
		vkof::device_wait_idle();
	}
	void test_case_exception(const doctest::TestCaseException &) override {}
	void subcase_start(const doctest::SubcaseSignature &) override {}
	void subcase_end() override {}
	void log_assert(const doctest::AssertData &) override {}
	void log_message(const doctest::MessageData &) override {}
	void test_case_skipped(const doctest::TestCaseData &) override {}
};
DOCTEST_REGISTER_LISTENER("device_idle_listener", 1, DeviceIdleListener);

extern "C" const char * __lsan_default_suppressions() {
	return (
		"leak:dbus_bus_register\n"
		"leak:vkof::tlas_build\n"
	);
}

int main(int argc, char ** argv) {
	bool windowed = false;
	bool timed = false;
	std::vector<char *> filtered;
	filtered.push_back(argv[0]);
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--windowed") == 0) {
			windowed = true;
		} else if (std::strcmp(argv[i], "--timed") == 0) {
			timed = true;
		} else {
			filtered.push_back(argv[i]);
		}
	}
	int filteredArgc = static_cast<int>(filtered.size());

	if (windowed) {
		vkof::init();
	} else {
		vkof::init_headless();
	}

	doctest::Context ctx(filteredArgc, filtered.data());

	// Print per-test-case runtime (doctest built-in duration reporting)
	if (timed) {
		ctx.setOption("duration", true);
	}

	// Exclude [windowed] test-suite when running headless
	if (!windowed) {
		ctx.addFilter("test-suite-exclude", "[windowed]");
	}

	int const result = ctx.run();
	printf("doctest result: %d\n", result);

	vkof::shutdown();
	return result;
}
