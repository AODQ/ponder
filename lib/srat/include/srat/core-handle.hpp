#pragma once

#include <srat/core-types.hpp>
#include <srat/alloc-virtual-range.hpp>

#include <source_location>
#include <type_traits>

#if SRAT_DEBUG()
#include <unordered_map>
#endif

// handle system for internal resource management

namespace srat
{

// pool of handles to manage resources
template <typename Handle, typename InternalResource>
struct HandlePool
{
	static_assert(
		std::is_same_v<decltype(Handle::id), u64>,
		"handle must have an id of type uint64_t"
	);

	static HandlePool<Handle, InternalResource> create(
		u32 const maxHandles,
		char const * debugName = "HandlePool"
	);
	~HandlePool();
	HandlePool(HandlePool const &) = delete;
	HandlePool & operator=(HandlePool const &) = delete;
	HandlePool(HandlePool &&);
	HandlePool & operator=(HandlePool &&);

	Handle allocate(
		InternalResource const & resource,
		std::source_location loc = std::source_location::current()
	);
	Handle allocate(
		InternalResource && resource,
		std::source_location loc = std::source_location::current()
	);
	void free(Handle const & handle);
	bool valid(Handle const & handle) const;
	InternalResource * get(Handle const & handle);

	[[nodiscard]] bool empty() const { return allocator.empty(); }

private:
	HandlePool(
		AllocVirtualRange && allocator,
		InternalResource * resourceAllocations,
		u32 const maxHandles
	)
		: allocator(std::move(allocator))
		, maxHandles(maxHandles)
		, resourceAllocations(resourceAllocations)
	{}

	void deleteInternal() const;

	AllocVirtualRange allocator;
	u32 maxHandles;
	InternalResource * resourceAllocations;

	// this is the handle pool gen, there is also a per-handle gen as part of
	//   its virtual range block

#if SRAT_DEBUG()
	std::unordered_map<u64, std::source_location> locationMap;
#endif
};

inline u64 handle_make(u32 const index, u32 const generation) {
	return (static_cast<u64>(generation) << 32) | index;
}
inline u32 handle_index(u64 const handle_id) { return handle_id & 0xFFFFFFFF; }
inline void handle_index_set(u64 & handle_id, u32 const index) {
	handle_id = (handle_id & 0xFFFFFFFF00000000) | index;
}
inline void handle_index_clear(u64 & handle_id) {
	handle_id &= 0xFFFFFFFF00000000;
}
inline u32 handle_generation(u64 const handle_id) {
	return (handle_id >> 32) & 0xFFFFFFFF;
}
inline void handle_generation_inc(u64 & handle_id) {
	if (handle_generation(handle_id) == UINT32_MAX) {
		// wrap around from max->0
		handle_id &= 0x00000000FFFFFFFF;
	}
	handle_id += (1ull << 32);
}
inline void handle_generation_clear(u64 & handle_id) {
	handle_id &= 0x00000000FFFFFFFF;
}
inline bool generation_alive(u32 const generation) {
	// odd=allocated, even=free,
	// 0=invalid
	return (generation != 0) && ((generation & 1) == 1);
}
inline u32 generation_inc(u32 & generation) {
	generation += 1;
	if (generation == 0) { generation = 1; }
	return generation;
}

} // namespace srat

#include <srat/core-handle.inl>
