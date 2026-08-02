// gpu side of the slang test harness (cpp side: src/util-slang.hpp)
//
// main must be [numthreads(skTestGroupSize, 1, 1)] and start with
//   if (!t_begin(tid.x)) { return; }
//
//   T_CHECK(cond)
//   T_CHECK_NEAR(actual, expected, tol)  -- float or floatN
//   T_PRINT(...)                         -- printf from thread 0 only
//   t_user_va()                          -- the test's own uploaded data
//
// printf output shows on stderr as "[probe] ..."

static const uint skTestGroupSize = 64u;

struct TestPush {
	// root constants occupy [0,128)
	uint4 rootPad[8];
	uint64_t resultVa;
	// test's own data, whatever the c++ side uploaded; 0 when unused
	uint64_t userVa;
	uint failureCapacity;
	uint threadCount;
	uint seed;
	uint pad;
};

[[vk::push_constant]] TestPush tPush;

// mirrored by SlangTestFailure
struct TestFailure {
	uint line;
	uint thread;
	uint componentCount;
	float tolerance;
	float4 actual;
	float4 expected;
};

struct TestResults {
	uint checkCount;
	uint failCount;
	uint pad0;
	uint pad1;
	// followed by failureCapacity TestFailure records
};

static uint tThread;

bool t_begin(const uint thread) {
	tThread = thread;
	return thread < tPush.threadCount;
}

uint t_thread() {
	return tThread;
}

// device address of SlangTestInfo::userVa, to be cast to the test's own type
uint64_t t_user_va() {
	return tPush.userVa;
}

void t_record(
	const uint line,
	const bool passed,
	const uint componentCount,
	const float4 actual,
	const float4 expected,
	const float tolerance
) {
	TestResults * results = (TestResults *)tPush.resultVa;
	InterlockedAdd(results->checkCount, 1u);
	if (passed) { return; }
	uint slot;
	InterlockedAdd(results->failCount, 1u, slot);
	if (slot >= tPush.failureCapacity) { return; }
	TestFailure * failures = (TestFailure *)(results + 1);
	failures[slot] = {
		line, tThread, componentCount, tolerance, actual, expected,
	};
}

// absolute below 1, relative above; nan fails
bool t_near(const float actual, const float expected, const float tol) {
	return abs(actual - expected) <= tol * max(1.0, abs(expected));
}

void t_check(const uint line, const bool cond) {
	t_record(line, cond, 0u, float4(0.0), float4(0.0), 0.0);
}

void t_check_near<let N : int>(
	const uint line,
	const vector<float, N> actual,
	const vector<float, N> expected,
	const float tol
) {
	bool passed = true;
	float4 a = float4(0.0);
	float4 e = float4(0.0);
	[ForceUnroll]
	for (int i = 0; i < N; ++i) {
		passed = passed && t_near(actual[i], expected[i], tol);
		a[i] = actual[i];
		e[i] = expected[i];
	}
	t_record(line, passed, uint(N), a, e, tol);
}

void t_check_near(
	const uint line, const float actual, const float expected, const float tol
) {
	t_check_near(line, vector<float, 1>(actual), vector<float, 1>(expected), tol);
}

#define T_CHECK(cond) t_check(__LINE__, (cond))
#define T_CHECK_NEAR(actual, expected, tol) t_check_near(__LINE__, (actual), (expected), (tol))
#define T_PRINT(...) if (tThread == 0u) { printf(__VA_ARGS__); }

// -- per-thread rng, seeded by thread index and push seed

uint t_pcg(const uint v) {
	const uint state = v * 747796405u + 2891336453u;
	const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

struct TestRng {
	uint state;

	[mutating]
	float next() {
		state = t_pcg(state);
		return float(state >> 8u) * (1.0 / 16777216.0);
	}

	[mutating]
	float range(const float lo, const float hi) {
		return lo + (hi - lo) * next();
	}

	[mutating]
	float3 range3(const float lo, const float hi) {
		return float3(range(lo, hi), range(lo, hi), range(lo, hi));
	}
};

TestRng t_rng() {
	return { t_pcg(tThread ^ t_pcg(tPush.seed)) };
}
