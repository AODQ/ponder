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
