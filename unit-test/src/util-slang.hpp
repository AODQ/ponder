#pragma once

// ---------------------------------------------------------------------------
// cpu side of the slang test harness (gpu side: shaders/slang/util-test.slang)
// ---------------------------------------------------------------------------

#include "util.hpp"

#include <cstdio>
#include <string>

namespace test {

// mirrors TestFailure in util-test.slang
struct SlangTestFailure {
	u32 line;
	u32 thread;
	u32 componentCount;
	f32 tolerance;
	f32 actual[4];
	f32 expected[4];
};
static_assert(sizeof(SlangTestFailure) == 48u);

// mirrors TestPush in util-test.slang, minus the root-constant pad
struct SlangTestPush {
	u64 resultVa;
	u64 userVa;
	u32 failureCapacity;
	u32 threadCount;
	u32 seed;
	u32 pad;
};
static_assert(sizeof(SlangTestPush) == 32u);

// matches skTestGroupSize in util-test.slang
static constexpr u32 skSlangTestGroupSize = 64u;

struct SlangTestInfo {
	// relative to shaders/slang/
	char const * file;
	u32 threadCount = 1u;
	u32 seed = 0u;
	u32 failureCapacity = 64u;
	srat::slice<char const * const> defines {};
	// resolves `import` as well as `#include`
	srat::slice<char const * const> includePaths {};
	// device address of the test's own data, read in the shader with
	// t_user_va(); the test owns the buffer's lifetime
	u64 userVa = 0u;
	// false to inspect failures yourself instead of failing the test case
	bool reportFailures = true;
};

struct SlangTestResult {
	bool compiled;
	u32 checkCount;
	u32 failCount;
	// at most failureCapacity of them
	std::vector<SlangTestFailure> failures;
};

inline std::string slang_test_format_values(
	f32 const * const values, u32 const componentCount
) {
	std::string out = "(";
	for (u32 i = 0u; i < componentCount; ++i) {
		out += (i == 0u ? "" : ", ") + std::to_string(values[i]);
	}
	return out + ")";
}

// vkof returns a valid handle on compile failure (for hot reload), so check
// compilation up front. returns "" on success
inline std::string slang_test_compile_errors(
	std::string const & path,
	srat::slice<char const * const> const defines,
	srat::slice<char const * const> const includePaths
) {
	std::string cmd = (
		"slangc -target spirv -stage compute -entry main"
		" -matrix-layout-column-major -force-glsl-scalar-layout"
		" -o " + std::string(TEST_SHADER_DIR "bin/slang-check.spv")
	);
	for (char const * const d : defines) {
		cmd += std::string(" -D") + d;
	}
	for (char const * const p : includePaths) {
		cmd += std::string(" -I") + p;
	}
	cmd += " " + path + " 2>&1";
	std::string output;
	FILE * const pipe = popen(cmd.c_str(), "r");
	if (!pipe) {
		return "could not run slangc";
	}
	char chunk[512];
	while (fgets(chunk, sizeof(chunk), pipe)) {
		output += chunk;
	}
	int const status = pclose(pipe);
	if (status == 0) {
		return "";
	}
	return output.empty() ? "slangc failed with no output" : output;
}

inline SlangTestResult slang_test_run(SlangTestInfo const & info) {
	std::string const path = std::string(TEST_SHADER_DIR "slang/") + info.file;
	std::string const errors = slang_test_compile_errors(
		path, info.defines, info.includePaths
	);
	if (!errors.empty()) {
		FAIL_CHECK("slang test failed to compile: ", path, "\n", errors);
		return {
			.compiled = false, .checkCount = 0u, .failCount = 0u, .failures = {},
		};
	}
	vkof::Pipeline const pipeline = vkof::pipeline_compute_create({
		.pathCompute = path.c_str(),
		.defines = info.defines,
		.includePaths = info.includePaths,
	});
	REQUIRE(pipeline.id != 0u);

	u32 const headerWords = 4u;
	u32 const failureWords = (
		info.failureCapacity * sizeof(SlangTestFailure) / sizeof(u32)
	);
	u32 const totalWords = headerWords + failureWords;
	vkof::Buffer const buffer = make_buffer_u32(totalWords);
	std::vector<u32> const zeros(totalWords, 0u);
	vkof::buffer_upload({
		.buffer = buffer,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(zeros.data()),
			static_cast<u64>(totalWords) * sizeof(u32)
		),
	});

	SlangTestPush const push {
		.resultVa = vkof::buffer_virtual_address(buffer),
		.userVa = info.userVa,
		.failureCapacity = info.failureCapacity,
		.threadCount = info.threadCount,
		.seed = info.seed,
		.pad = 0u,
	};
	u32 const groupCount = (
		(info.threadCount + skSlangTestGroupSize - 1u) / skSlangTestGroupSize
	);
	dispatch(pipeline, push, groupCount);
	std::vector<u32> const words = readback<u32>(buffer, 0u, totalWords);
	vkof::buffer_destroy(buffer);
	vkof::pipeline_destroy(pipeline);

	SlangTestResult result {
		.compiled = true,
		.checkCount = words[0],
		.failCount = words[1],
		.failures = {},
	};
	u32 const storedCount = std::min(result.failCount, info.failureCapacity);
	result.failures.resize(storedCount);
	std::memcpy(
		result.failures.data(),
		words.data() + headerWords,
		storedCount * sizeof(SlangTestFailure)
	);

	if (!info.reportFailures) {
		return result;
	}
	if (result.checkCount == 0u) {
		FAIL_CHECK("slang test ran no checks: ", path);
	}
	for (SlangTestFailure const & f : result.failures) {
		std::string const message = (
			f.componentCount == 0u
			? "T_CHECK failed [thread " + std::to_string(f.thread) + "]"
			: (
				"T_CHECK_NEAR failed [thread " + std::to_string(f.thread) + "]"
				+ "\n  actual:   "
				+ slang_test_format_values(f.actual, f.componentCount)
				+ "\n  expected: "
				+ slang_test_format_values(f.expected, f.componentCount)
				+ "\n  tol:      " + std::to_string(f.tolerance)
			)
		);
		ADD_FAIL_CHECK_AT(path.c_str(), static_cast<int>(f.line), message);
	}
	if (result.failCount > storedCount) {
		FAIL_CHECK(
			std::to_string(result.failCount - storedCount),
			" more slang check failures not stored (raise failureCapacity)"
		);
	}
	return result;
}

} // namespace test
