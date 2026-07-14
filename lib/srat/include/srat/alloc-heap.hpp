#pragma once

#include <srat/core-config.hpp>
#include <srat/core-types.hpp>

#include <source_location>
#include <utility>

#if SRAT_DEBUG()
#include <atomic>
#include <mutex>
#include <unordered_map>
#endif

namespace srat {

#if SRAT_DEBUG()
namespace detail {
	struct HeapTracker {
		std::mutex lock;
		std::unordered_map<void const *, std::source_location> allocs;
	};
	inline HeapTracker & heap_tracker() {
		static HeapTracker sTracker;
		return sTracker;
	}
}
#endif

template <typename T>
[[nodiscard]] T * heap_alloc(
	T && data,
	std::source_location const loc = std::source_location::current()
) {
	T * const ptr = new T(std::forward<T>(data));
#if SRAT_DEBUG()
	{
		auto & tracker = detail::heap_tracker();
		std::lock_guard const guard { tracker.lock };
		tracker.allocs.emplace(ptr, loc);
	}
#endif
	return ptr;
}

template <typename T>
void heap_free(T * const ptr) {
#if SRAT_DEBUG()
	if (ptr) {
		auto & tracker = detail::heap_tracker();
		std::lock_guard const guard { tracker.lock };
		tracker.allocs.erase(ptr);
	}
#endif
	delete ptr;
}

// Call at shutdown after all threads have joined.
// In debug builds, prints any remaining allocations and asserts zero leaks.
inline void heap_assert_no_leaks() {
#if SRAT_DEBUG()
	auto & tracker = detail::heap_tracker();
	std::lock_guard const guard { tracker.lock };
	for (auto const & [ptr, loc] : tracker.allocs) {
		printf(
			"leak: %p allocated at %s:%u\n",
			ptr, loc.file_name(), loc.line()
		);
	}
	SRAT_ASSERT(tracker.allocs.empty());
#endif
}

} // namespace srat
