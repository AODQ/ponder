#pragma once

#include <chrono>
#include <cstdio>

namespace srat {

struct profile_tick {
	using Clock = std::chrono::steady_clock;
	using Ms = std::chrono::duration<double, std::milli>;

	Clock::time_point t = Clock::now();

	void tick(char const * label) {
		printf("  [%.1f ms] %s\n", Ms(Clock::now() - t).count(), label);
		t = Clock::now();
	}
};

} // namespace srat
