// self-test of util-test.slang, and the smallest possible autodiff examples:
// derivatives you can check by hand against fwd_diff's answer

#include "util-test.slang"

[Differentiable]
float square(const float x) {
	return x * x;
}

[Differentiable]
float sin_of_square(const float x) {
	return sin(square(x));
}

[shader("compute")]
[numthreads(skTestGroupSize, 1, 1)]
void main(const uint3 tid : SV_DispatchThreadID) {
	if (!t_begin(tid.x)) { return; }
	TestRng rng = t_rng();
	let x = rng.range(-2.0, 2.0);

	T_PRINT("harness: thread 0 x = %f\n", x);

	// -- d/dx x^2 = 2x. diffPair(x, 1.0) seeds dx = 1, so .d is df/dx
	let sq = fwd_diff(square)(diffPair(x, 1.0));
	T_CHECK_NEAR(sq.p, x * x, 1e-6);
	T_CHECK_NEAR(sq.d, 2.0 * x, 1e-6);

	// -- chain rule through a call: d/dx sin(x^2) = cos(x^2) * 2x
	let ss = fwd_diff(sin_of_square)(diffPair(x, 1.0));
	T_CHECK_NEAR(ss.d, cos(x * x) * 2.0 * x, 1e-5);

	T_CHECK_NEAR(float2(x, 1.0), float2(x, 1.0), 0.0);
	T_CHECK(t_thread() == tid.x);

#if defined(HARNESS_EXPECT_FAIL)
	// the cpp side checks that exactly this one fails, on this line
	T_CHECK_NEAR(float3(1.0, 2.0, 3.0), float3(1.0, 2.5, 3.0), 1e-3);
#endif
}
