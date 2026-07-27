#pragma once

#include <srat/core-array.hpp>
#include <srat/core-types.hpp>

// virtual range allocator
// allocate virtual memory ranges, user has to provide actual backing memory
// the 'element' can be bytes, pages, or any other unit of memory.

namespace srat {

struct AllocVirtualRange;

enum struct AllocVirtualRangeStrategy : u64 {
	FreeList, // default strategy
	Linear, // simple linear allocator for fast clear
};

struct AllocVirtualRangeBlock
{
	u64 elementCount;
	u64 elementOffset;
	u64 handle; // opaque handle to identify the allocated block

	// this will be invalid if the block was not allocated or has been freed
	[[nodiscard]] bool valid(AllocVirtualRange const & allocator) const;
};

struct AllocVirtualRangeParams
{
	u64 elementCount { 0 };
	u64 elementAlignment = 1u;
};

struct AllocVirtualRangeCreateParams
{
	char const * debugName;
	u64 elementCount;
	// only used for free-list strategy
	u32 maxBlockAllocations;
	AllocVirtualRangeStrategy strategy;
};

struct AllocVirtualRange
{
	// allocate a range of elements
	AllocVirtualRangeBlock allocate(AllocVirtualRangeParams const & request);

	// free a previously allocated range of elements
	void free(u64 const handle);

	// reset allocator to initial data, freeing all virtual memory ranges
	void clear();

	// create a new virtual range allocator
	static AllocVirtualRange create(AllocVirtualRangeCreateParams const & params);

	// this returns if the index has data in its slot
	[[nodiscard]] bool isIndexAlive(u32 blockIndex) const;
	// this returns if the handle is alive
	[[nodiscard]] bool isHandleAlive(u64 const handle) const;
	[[nodiscard]] u64 elementOffset(u32 blockIndex) const;

	// check if the allocator has no allocated blocks
	[[nodiscard]] bool empty() const;

	// this returns the total capacity of the allocator in elements
	[[nodiscard]] u64 capacity() const;

	// only valid for linear strategy,
	// this returns the number of allocated elements
	[[nodiscard]] u64 allocatedCount() const;

#if SRAT_DEBUG()
	void printAllocationStats() const;
	[[nodiscard]] char const * debugName() const;
#endif

	// virtual range allocators can't be copied
	~AllocVirtualRange();
	AllocVirtualRange(AllocVirtualRange const &) = delete;
	AllocVirtualRange & operator=(AllocVirtualRange const &) = delete;
	AllocVirtualRange(AllocVirtualRange &&);
	AllocVirtualRange & operator=(AllocVirtualRange &&);

private:
	// private constructor to enforce creation through the static create method
	AllocVirtualRange() = default;

	void moveFrom(AllocVirtualRange & o);

	friend struct AllocVirtualRangeBlock;

public:
	// internal data for the allocator
	srat::array<u64, 12> _internalData {};
};

// this verifies all allocators are empty, used at program exit for leaks
#if SRAT_DEBUG()
bool alloc_virtual_range_all_empty();
void alloc_virtual_range_verify_all_empty();
#endif

} // namespace srat
