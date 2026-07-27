#include <srat/alloc-virtual-range.hpp>

#include <srat/core-handle.hpp>

#include <set> // only for tracking allocators

// -----------------------------------------------------------------------------
// -- virtual range internal data structure
// -----------------------------------------------------------------------------

// TODO: remove the odd/even generation for alive/dead tracking, i already
//       use an allocated bool so it's a bit redundant
//       But this is super minor because I feel this is already very well
//       tested

namespace
{

struct AllocImplVirtualRangeBlock
{
	u64 elementOffset;
	u64 elementCount;
	u32 nextFreeIndex;
	u32 generation;
	bool allocated;
};

// just for internal data structure to keep an array
template <typename T> struct AllocImplPtrArray {
	T * data { nullptr };
	size_t elemCount { 0 };

	static AllocImplPtrArray create(size_t count) {
		AllocImplPtrArray arr {};
		arr.data = new T[count]; //NOLINT
		arr.elemCount = count;
		return arr;
	}
	void free() {
		delete[] data;
		data = nullptr;
		elemCount = 0;
	}

	[[nodiscard]] operator srat::slice<T>() { return { data, elemCount }; }
	[[nodiscard]] operator srat::slice<T>() const { return { data, elemCount }; }

	[[nodiscard]] T & operator[](size_t i) {
		SRAT_ASSERT(i < elemCount);
		return data[i]; //NOLINT
	}
	[[nodiscard]] T const & operator[](size_t i) const {
		SRAT_ASSERT(i < elemCount);
		return data[i]; //NOLINT
	}
};

struct AllocImplVirtualRangeDataFreeList
{
	u32 maxBlockAllocations;
	AllocImplPtrArray<AllocImplVirtualRangeBlock> allocatedBlocks;
	u32 freeListHeadIndex { 0u };
};

struct AllocImplVirtualRangeDataLinear
{
	u32 nextIndex;
};

struct AllocImplVirtualRangeData
{
	char const * debugName { nullptr };
	u64 elementCount { 0 };
	union {
		AllocImplVirtualRangeDataFreeList freelist;
		AllocImplVirtualRangeDataLinear linear {};
	};
	srat::AllocVirtualRangeStrategy strategy {};

	[[nodiscard]] bool isAlive(u32 index) const
	{
		u32 const gen = freelist.allocatedBlocks[index].generation;
		bool const allocated = freelist.allocatedBlocks[index].allocated;
		return gen != 0 && srat::generation_alive(gen) && allocated;
	}

	[[nodiscard]] bool isDead(u32 index) const
	{
		u32 const gen = freelist.allocatedBlocks[index].generation;
		bool const allocated = freelist.allocatedBlocks[index].allocated;
		return gen == 0 || (!srat::generation_alive(gen) && !allocated);
	}

	void sortFreeList();
};

static_assert(
	sizeof(AllocImplVirtualRangeData) <= sizeof(srat::AllocVirtualRange),
	"fit mismatch"
);

static constexpr u32 skFreeListEnd = UINT32_MAX;

} // namespace

static AllocImplVirtualRangeData & AllocatorData(
	srat::AllocVirtualRange & allocator
) {
	return (
		*reinterpret_cast<AllocImplVirtualRangeData *>(
			allocator._internalData.ptr()
		)
	);
}
static AllocImplVirtualRangeData const & AllocatorDataConst(
	srat::AllocVirtualRange const & allocator
) {
	return (
		*reinterpret_cast<AllocImplVirtualRangeData const *>(
			allocator._internalData.ptr()
		)
	);
}

#if SRAT_DEBUG()
static std::set<srat::AllocVirtualRange *> & allocators() {
	static std::set<srat::AllocVirtualRange *> sAllocators;
	return sAllocators;
}
#endif

void AllocImplVirtualRangeData::sortFreeList()
{
	// -- sort the free list by element offset using insertion sort
	u32 sortedHead = skFreeListEnd;
	u32 it = freelist.freeListHeadIndex;
	while (it != skFreeListEnd) {
		auto & currBlock = freelist.allocatedBlocks[it];
		Let nextIt = u32 { currBlock.nextFreeIndex };
		// insert current block into sorted list
		if (
			   sortedHead == skFreeListEnd
			|| (
				currBlock.elementOffset
				< freelist.allocatedBlocks[sortedHead].elementOffset
			)
		) {
			// insert at head of sorted list
			currBlock.nextFreeIndex = sortedHead;
			sortedHead = it;
			it = nextIt;
			continue;
		}
		// insert into middle or end of sorted list
		u32 sortedIt = sortedHead;
		while (true) {
			auto & sortedBlock = freelist.allocatedBlocks[sortedIt];
			Let sortedNext = u32 { sortedBlock.nextFreeIndex };
			if (
					sortedNext == skFreeListEnd
				|| (
					  currBlock.elementOffset
					< freelist.allocatedBlocks[sortedNext].elementOffset
				)
			) {
				// insert after sortedIt
				currBlock.nextFreeIndex = sortedNext;
				sortedBlock.nextFreeIndex = it;
				break;
			}
			sortedIt = sortedNext;
		}
		it = nextIt;
	}
	freelist.freeListHeadIndex = sortedHead;
}

// -----------------------------------------------------------------------------
// -- virtual range block implementation
// -----------------------------------------------------------------------------

bool srat::AllocVirtualRangeBlock::valid(
    AllocVirtualRange const & allocator
 ) const
 {
	u32 const idx = handle_index(this->handle);
	AllocImplVirtualRangeData const & self = AllocatorDataConst(allocator);
	SRAT_ASSERT(idx < self.freelist.maxBlockAllocations);
	Let block = self.freelist.allocatedBlocks[idx];
	if (!block.allocated) { return false; }
	return handle_generation(this->handle) == block.generation;
 }

// -----------------------------------------------------------------------------
// -- virtual range allocator interface
// -----------------------------------------------------------------------------

srat::AllocVirtualRange srat::AllocVirtualRange::create(
	AllocVirtualRangeCreateParams const & params
)
{
	SRAT_ASSERT(params.elementCount > 0);
	AllocVirtualRange allocator {};
	AllocImplVirtualRangeData & self = AllocatorData(allocator);
	// initialize the allocator with the provided parameters
	self = {
		.debugName = nullptr,
		.elementCount = params.elementCount,
		.freelist = {},
		.strategy = params.strategy,
	};
	switch (params.strategy) {
		case AllocVirtualRangeStrategy::FreeList: {
			SRAT_ASSERT(params.maxBlockAllocations > 0);
			SRAT_ASSERT(params.maxBlockAllocations + 1 < skFreeListEnd);
			// note; need extra block for free block otherwise would run out of
			//       memory when user allocates the max block allocations
			self.freelist = {
				.maxBlockAllocations = params.maxBlockAllocations + 1,
				.allocatedBlocks = (
					AllocImplPtrArray<AllocImplVirtualRangeBlock>::create(
						params.maxBlockAllocations + 1
					)
				),
				.freeListHeadIndex = 0,
			};
			self.freelist.freeListHeadIndex = 0;
			for (u32 it = 0; it < params.maxBlockAllocations + 1; ++it) {
				self.freelist.allocatedBlocks[it] = AllocImplVirtualRangeBlock {
					.elementOffset = 0u,
					.elementCount = 0u,
					.nextFreeIndex = skFreeListEnd,
					.generation = 0u, // invalid state
					.allocated = false,
				};
			}
			// initialize the allocated blocks to an empty state
			self.freelist.allocatedBlocks[0] = AllocImplVirtualRangeBlock {
				.elementOffset = 0,
				.elementCount = params.elementCount,
				.nextFreeIndex = skFreeListEnd,
				.generation = 1u, // initialize to free (odd=free)
				.allocated = false,
			};
		} break;
		case AllocVirtualRangeStrategy::Linear:
			self.linear.nextIndex = 0;
		break;
	}

#if SRAT_DEBUG()
	self.debugName = params.debugName; // TODO alloc+copy this
#endif
	return allocator;
}

bool srat::AllocVirtualRange::isIndexAlive(u32 blockIndex) const
{
	AllocImplVirtualRangeData const & self = AllocatorDataConst(*this);
	if (self.strategy == AllocVirtualRangeStrategy::Linear) {
		return blockIndex < self.linear.nextIndex;
	}
	SRAT_ASSERT(blockIndex < self.freelist.maxBlockAllocations);
	return self.isAlive(blockIndex);
}

bool srat::AllocVirtualRange::isHandleAlive(u64 const blockHandle) const
{
	AllocImplVirtualRangeData const & self = AllocatorDataConst(*this);
	if (self.strategy == AllocVirtualRangeStrategy::Linear) {
		return blockHandle < self.linear.nextIndex;
	}
	// recreate the VirtualRangeBlock
	Let block = AllocVirtualRangeBlock {
		.elementCount = 0, // not needed for validity check
		.elementOffset = 0, // not needed for validity check
		.handle = blockHandle,
	};
	return block.valid(*this) && self.isAlive(handle_index(blockHandle));
}

u64 srat::AllocVirtualRange::elementOffset(u32 const blockIndex) const
{
	AllocImplVirtualRangeData const & self = AllocatorDataConst(*this);
	if (self.strategy == AllocVirtualRangeStrategy::Linear) {
		return blockIndex; // in linear strategy, the block index is the offset
	}
	SRAT_ASSERT(self.isAlive(blockIndex));
	return self.freelist.allocatedBlocks[blockIndex].elementOffset;
}

void srat::AllocVirtualRange::moveFrom(AllocVirtualRange & o)
{
	// move the internal data from the source allocator to this allocator
	AllocImplVirtualRangeData & self = AllocatorData(*this);
	AllocImplVirtualRangeData & otherSelf = AllocatorData(o);

	// apply the move constructor to the internal debugging data
#if SRAT_DEBUG()
	{
		auto it = allocators().find(&o);
		// this can be either a move or from first creation
		if (it != allocators().end()) {
			allocators().erase(it);
		}
		allocators().insert(this);
	}
#endif

	// move the internal data
	self = otherSelf;
	// invalidate the source allocator's internal data
	otherSelf = AllocImplVirtualRangeData {
		.debugName = {},
		.elementCount = 0,
		.freelist = {},
	};
}

srat::AllocVirtualRange::AllocVirtualRange(AllocVirtualRange && o)
	: _internalData()
{
	moveFrom(o);
}

srat::AllocVirtualRange & srat::AllocVirtualRange::operator=(
	AllocVirtualRange && o
)
{
	if (this != &o) {
		// free existing resources
		AllocImplVirtualRangeData & self = AllocatorData(*this);
		if (self.strategy == AllocVirtualRangeStrategy::FreeList) {
			self.freelist.allocatedBlocks.free();
		}
		moveFrom(o);
	}
	return *this;
}

srat::AllocVirtualRange::~AllocVirtualRange()
{
	AllocImplVirtualRangeData & self = AllocatorData(*this);
	if (self.strategy == AllocVirtualRangeStrategy::FreeList) {
		self.freelist.allocatedBlocks.free();
	}
#if SRAT_DEBUG()
	Let it = allocators().find(this);
	if (it != allocators().end()) {
		allocators().erase(it);
	}
#endif
}

// -----------------------------------------------------------------------------
// -- virtual range allocator implementation linear
// -----------------------------------------------------------------------------

srat::AllocVirtualRangeBlock virtual_range_linear_allocate(
	AllocImplVirtualRangeData & self,
	srat::AllocVirtualRangeParams const & request
) {
	SRAT_ASSERT(request.elementCount < UINT32_MAX);
	SRAT_ASSERT(request.elementAlignment < UINT32_MAX);
	// bump up linear allocator by alignment requirement
	Let allocIndex = u32 {
		srat::alignUp(self.linear.nextIndex, (u32)request.elementAlignment)
	};
	if (allocIndex + request.elementCount > self.elementCount) {
		printf(
			"[warn] VirtualRangeAllocator '%s' out of memory: "
			"requested %zu elements with alignment %zu, "
			"however container is currently %zu/%zu\n",
			self.debugName,
			(size_t)request.elementCount,
			(size_t)request.elementAlignment,
			(size_t)allocIndex,
			(size_t)self.elementCount
		);
		// not enough space to satisfy allocation
		return srat::AllocVirtualRangeBlock {
			.elementCount = 0,
			.elementOffset = 0,
			.handle = 0,
		};
	}
	self.linear.nextIndex = allocIndex + (u32)request.elementCount;
	return srat::AllocVirtualRangeBlock {
		.elementCount = request.elementCount,
		.elementOffset = allocIndex,
		.handle = 0, // linear allocator doesn't support freeing so no handle
	};
}

void virtual_range_linear_free(
	[[maybe_unused]]AllocImplVirtualRangeData & self,
	[[maybe_unused]]u64 const handle
) {
	// linear allocator doesn't support freeing, so this is a no-op
}

void virtual_range_linear_clear(
	AllocImplVirtualRangeData & self
) {
	self.linear.nextIndex = 0;
}

bool virtual_range_linear_empty(
	AllocImplVirtualRangeData const & self
) {
	return self.linear.nextIndex == 0;
}

// -----------------------------------------------------------------------------
// -- virtual range allocator implementation free-list
// -----------------------------------------------------------------------------

srat::AllocVirtualRangeBlock virtual_range_freelist_allocate(
	AllocImplVirtualRangeData & self,
	srat::AllocVirtualRangeParams const & request
) {
	// -- walk the free list to find a block that can satisfy the request
	u32 prevFreeIndex = skFreeListEnd;
	u32 freeIndex = self.freelist.freeListHeadIndex;
	auto & freelist = self.freelist;
	while (freeIndex != skFreeListEnd) {
		// check if the current free block can satisfy the request
		auto & freeBlock = freelist.allocatedBlocks[freeIndex];
		Let alignedOffset = u64 {
			srat::alignUp(freeBlock.elementOffset, request.elementAlignment)
		};
		Let padding = u64 {alignedOffset - freeBlock.elementOffset};
		Let leadingFreeBlockOffset = u64 {freeBlock.elementOffset};
		if (freeBlock.elementCount < request.elementCount + padding) {
			// move to the next free block
			prevFreeIndex = freeIndex;
			freeIndex = freeBlock.nextFreeIndex;
			continue;
		}
		// -- found a free block that can satisfy allocation
		Let remainingCount = u64 {
			freeBlock.elementCount - request.elementCount - padding
		};
		Let allocatedOffset = u64 {alignedOffset};
		// -- update the free block to reflect the allocated range
		freeBlock.elementOffset += request.elementCount + padding;
		freeBlock.elementCount = remainingCount;
		if (freeBlock.elementCount == 0) {
			// remove the free block from the free list if it's fully allocated
			if (prevFreeIndex == skFreeListEnd) {
				freelist.freeListHeadIndex = freeBlock.nextFreeIndex;
			} else {
				freelist.allocatedBlocks[prevFreeIndex].nextFreeIndex = (
					freeBlock.nextFreeIndex
				);
			}
			// invalidate free block handle
			if (srat::generation_alive(freeBlock.generation)) {
				SRAT_ASSERT(freeBlock.allocated == false);
				srat::generation_inc(freeBlock.generation);
				SRAT_ASSERT(self.isDead(freeIndex));
			}
			freeBlock.elementOffset = 0;
			freeBlock.elementCount = 0;
			freeBlock.allocated = false;
			freeBlock.nextFreeIndex = skFreeListEnd;
		}
		// -- create a new allocated block for the allocated range
		auto allocatedIndex = 1u;
		for (; allocatedIndex < freelist.maxBlockAllocations; ++allocatedIndex) {
			if (self.isDead(allocatedIndex)) {
				break;
			}
		}
		SRAT_ASSERT(allocatedIndex < freelist.maxBlockAllocations);
		SRAT_ASSERT(self.isDead(allocatedIndex));
		freelist.allocatedBlocks[allocatedIndex] = AllocImplVirtualRangeBlock {
			.elementOffset = leadingFreeBlockOffset,
			.elementCount = request.elementCount + padding,
			.nextFreeIndex = skFreeListEnd,
			.generation = freelist.allocatedBlocks[allocatedIndex].generation,
			.allocated = true,
		};
		srat::generation_inc(freelist.allocatedBlocks[allocatedIndex].generation);
		freelist.allocatedBlocks[allocatedIndex].allocated = true;
		SRAT_ASSERT(self.isAlive(allocatedIndex));

		return srat::AllocVirtualRangeBlock {
			.elementCount = request.elementCount,
			.elementOffset = allocatedOffset,
			.handle = (
				srat::handle_make(
					allocatedIndex,
					freelist.allocatedBlocks[allocatedIndex].generation
				)
			),
		};
	}

	// no free block could satisfy the request
	return srat::AllocVirtualRangeBlock {
		.elementCount = 0,
		.elementOffset = 0,
		.handle = 0,
	};
}

void virtual_range_freelist_free(
	srat::AllocVirtualRange & selfAlloc,
	AllocImplVirtualRangeData & self,
	u64 const handle
) {
	auto & freelist = self.freelist;
	Let block = srat::AllocVirtualRangeBlock {
		.elementCount = 0, // not needed for free
		.elementOffset = 0, // not needed for free
		.handle = handle,
	};
	if (!block.valid(selfAlloc)) { return; }
	Let blockIndex = u32{srat::handle_index(block.handle)};
	auto & blockInternal = freelist.allocatedBlocks[blockIndex];

	// -- increment the generation to invalidate the block's handle
	// but need to increment twice to keep in an alive state
	srat::generation_inc(blockInternal.generation);
	srat::generation_inc(blockInternal.generation);
	SRAT_ASSERT(self.isAlive(blockIndex));
	blockInternal.allocated = false;

	// -- create a new free block for the freed range
	blockInternal.nextFreeIndex = freelist.freeListHeadIndex;

	// -- insert the new free block into the free list
	freelist.freeListHeadIndex = blockIndex;

	// -- keep free list sorted by element offset
	self.sortFreeList();

	// -- coalesce adjacent free blocks in the free list
	for (u32 it = freelist.freeListHeadIndex; it != skFreeListEnd;) {
		auto & currentBlock = freelist.allocatedBlocks[it];
		Let nextIndex = u32 {currentBlock.nextFreeIndex};
		if (nextIndex != skFreeListEnd) {
			auto & nextBlock = freelist.allocatedBlocks[nextIndex];
			if (
				currentBlock.elementOffset + currentBlock.elementCount ==
				nextBlock.elementOffset
			) {
				// coalesce current block with next block
				currentBlock.elementCount += nextBlock.elementCount;
				currentBlock.nextFreeIndex = nextBlock.nextFreeIndex;
				// invalidate the next block's handle by incrementing its generation
				if (srat::generation_alive(nextBlock.generation)) {
					SRAT_ASSERT(nextBlock.allocated == false);
					srat::generation_inc(nextBlock.generation);
					SRAT_ASSERT(self.isDead(nextIndex));
				}
				nextBlock.elementOffset = 0;
				nextBlock.elementCount = 0;
				nextBlock.nextFreeIndex = skFreeListEnd;
				nextBlock.allocated = false;
				continue; // check for further coalescing with the new next block
			}
		}
		it = nextIndex;
	}
}

void virtual_range_freelist_clear(
	AllocImplVirtualRangeData & self
) {
	auto & freelist = self.freelist;
	for (u32 it = 0; it < freelist.maxBlockAllocations; ++it) {
		// kill alive blocks by incrementing generation
		if (self.isAlive(it)) {
			srat::generation_inc(freelist.allocatedBlocks[it].generation);
			freelist.allocatedBlocks[it].allocated = false;
			SRAT_ASSERT(self.isDead(it));
		}
		// reset block to an empty state
		u32 const gen = freelist.allocatedBlocks[it].generation;
		freelist.allocatedBlocks[it] = AllocImplVirtualRangeBlock {
			.elementOffset = 0,
			.elementCount = 0,
			.nextFreeIndex = skFreeListEnd,
			.generation = gen, // preserve generation to invalidate handles
			.allocated = false,
		};
	}
	// reinitialize slot 0
	if (self.isDead(0)) { // resurrect as free block
		srat::generation_inc(freelist.allocatedBlocks[0].generation);
		freelist.allocatedBlocks[0].allocated = false;
	}
	freelist.allocatedBlocks[0].elementCount = self.elementCount;
	freelist.freeListHeadIndex = 0u;
}

bool virtual_range_freelist_empty(
	AllocImplVirtualRangeData const & self
) {
	Let freelist = self.freelist;
	Let freeHead = u32 {freelist.freeListHeadIndex};
	if (freeHead == skFreeListEnd) { return false; }
	AllocImplVirtualRangeBlock const & firstBlock = (
		freelist.allocatedBlocks[freeHead]
	);

	return (
		   firstBlock.elementCount == self.elementCount
		&& firstBlock.elementOffset == 0
		&& (firstBlock.generation&1) == 1 // free block
		&& firstBlock.nextFreeIndex == skFreeListEnd
		&& !firstBlock.allocated
	);
}

// -----------------------------------------------------------------------------
// -- virtual range allocator implementation interface
// -----------------------------------------------------------------------------

srat::AllocVirtualRangeBlock srat::AllocVirtualRange::allocate(
	AllocVirtualRangeParams const & request
) {
	auto & self = AllocatorData(*this);
	switch (self.strategy) {
		case AllocVirtualRangeStrategy::FreeList:
			return virtual_range_freelist_allocate(self, request);
		case AllocVirtualRangeStrategy::Linear:
			return virtual_range_linear_allocate(self, request);
	}
	exit(1); // should never reach here
}

void srat::AllocVirtualRange::free(u64 const handle)
{
	auto & self = AllocatorData(*this);
	switch (self.strategy) {
		case AllocVirtualRangeStrategy::FreeList:
			return virtual_range_freelist_free(*this, self, handle);
		case AllocVirtualRangeStrategy::Linear:
			return virtual_range_linear_free(self, handle);
	}
}

void srat::AllocVirtualRange::clear()
{
	auto & self = AllocatorData(*this);
	switch (self.strategy) {
		case AllocVirtualRangeStrategy::FreeList:
			return virtual_range_freelist_clear(self);
		case AllocVirtualRangeStrategy::Linear:
			return virtual_range_linear_clear(self);
	}
}

#if SRAT_DEBUG()
void srat::AllocVirtualRange::printAllocationStats() const
{
}

char const * srat::AllocVirtualRange::debugName() const
{
	return AllocatorDataConst(*this).debugName;
}
#endif

bool srat::AllocVirtualRange::empty() const
{
	Let self = AllocatorDataConst(*this);
	switch (self.strategy) {
		case AllocVirtualRangeStrategy::FreeList:
			return virtual_range_freelist_empty(self);
		case AllocVirtualRangeStrategy::Linear:
			return virtual_range_linear_empty(self);
	}
	exit(1); // should never reach here
}

u64 srat::AllocVirtualRange::capacity() const
{
	return AllocatorDataConst(*this).elementCount;
}

u64 srat::AllocVirtualRange::allocatedCount() const
{
	Let self = AllocatorDataConst(*this);
	SRAT_ASSERT(self.strategy == AllocVirtualRangeStrategy::Linear);
	return self.linear.nextIndex;
}

#if SRAT_DEBUG()
bool srat::alloc_virtual_range_all_empty()
{
	for (auto allocator : allocators()) {
		if (!allocator->empty()) {
			printf("allocator '%s' not empty\n", allocator->debugName());
			return false;
		}
	}
	return true;
}

void srat::alloc_virtual_range_verify_all_empty()
{
	for (auto allocator : allocators()) {
		if (!allocator->empty()) {
			printf(
				"allocator '%s' not empty\n",
				allocator->debugName()
			);
			SRAT_ASSERT(false && "allocator not empty");
		}
	}
}
#endif
