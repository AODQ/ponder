#pragma once
#include <srat/core-handle.hpp>

#include <new> // for std::align_val_t

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)

template <typename Handle, typename InternalResource>
srat::HandlePool<Handle, InternalResource>
srat::HandlePool<Handle, InternalResource>::create(
	u32 const maxHandles,
	char const * debugName
)
{
	InternalResource * const resourceAllocations = (
		static_cast<InternalResource *>(
			operator new[](
				sizeof(InternalResource) * maxHandles,
				std::align_val_t{alignof(InternalResource)}
			)
		)
	);
	return srat::HandlePool<Handle, InternalResource>(
		AllocVirtualRange::create(
			AllocVirtualRangeCreateParams {
				.debugName = debugName,
				.elementCount = maxHandles,
				.maxBlockAllocations = maxHandles,
				.strategy = AllocVirtualRangeStrategy::FreeList,
			}
		),
		resourceAllocations,
		maxHandles
	);
}

template <typename Handle, typename InternalResource>
void srat::HandlePool<Handle, InternalResource>::deleteInternal() const {
	if (resourceAllocations == nullptr) { return; }
#if SRAT_DEBUG()
	for (auto const & [id, loc] : locationMap) {
		printf(
			"handle leak: id=0x%llx allocated at %s:%u\n",
			(unsigned long long)id, loc.file_name(), loc.line()
		);
	}
	SRAT_ASSERT(locationMap.empty());
#endif
	for (u32 i = 0; i < maxHandles; ++i) {
		if (allocator.isIndexAlive(i)) {
			u64 const elementOffset = allocator.elementOffset(i);
			SRAT_ASSERT(elementOffset < maxHandles);
			resourceAllocations[elementOffset].~InternalResource();
		}
	}
	operator delete[](
		resourceAllocations,
		std::align_val_t{alignof(InternalResource)}
	);
}

template <typename Handle, typename InternalResource>
srat::HandlePool<Handle, InternalResource>::HandlePool::~HandlePool() {
	deleteInternal();
}

template <typename Handle, typename InternalResource>
srat::HandlePool<Handle, InternalResource>::HandlePool(HandlePool && o)
	: allocator(std::move(o.allocator))
	, maxHandles(o.maxHandles)
	, resourceAllocations(o.resourceAllocations)
{
	o.resourceAllocations = nullptr;
}

template <typename Handle, typename InternalResource>
srat::HandlePool<Handle, InternalResource> &
srat::HandlePool<Handle, InternalResource>::operator=(HandlePool && o)
{
	if (this != &o) {
		deleteInternal();
		allocator = std::move(o.allocator);
		resourceAllocations = o.resourceAllocations;
		maxHandles = o.maxHandles;
		o.resourceAllocations = nullptr;
		o.maxHandles = 0;
	}
	return *this;
}

template <typename Handle, typename InternalResource>
Handle srat::HandlePool<Handle, InternalResource>::allocate(
	InternalResource const & resource,
	std::source_location const loc
) {
	srat::AllocVirtualRangeBlock const handleId = (
		allocator.allocate({.elementCount = 1})
	);
	auto const elementOffset = handleId.elementOffset;
	SRAT_ASSERT(elementOffset < maxHandles);
	if (!handleId.valid(allocator)) {
		return Handle { 0 };
	}
	new (&resourceAllocations[elementOffset]) InternalResource(resource);
#if SRAT_DEBUG()
	locationMap.emplace(handleId.handle, loc);
#endif
	return Handle { handleId.handle };
}

template <typename Handle, typename InternalResource>
Handle srat::HandlePool<Handle, InternalResource>::allocate(
	InternalResource && resource,
	std::source_location const loc
) {
	srat::AllocVirtualRangeBlock const handleId = (
		allocator.allocate({.elementCount = 1})
	);
	auto const elementOffset = handleId.elementOffset;
	SRAT_ASSERT(elementOffset < maxHandles);
	if (!handleId.valid(allocator)) {
		return Handle { 0 };
	}
	new (&resourceAllocations[elementOffset])
		InternalResource(std::move(resource));
#if SRAT_DEBUG()
	locationMap.emplace(handleId.handle, loc);
#endif
	return Handle { handleId.handle };
}

template <typename Handle, typename InternalResource>
void srat::HandlePool<Handle, InternalResource>::free(Handle const & handle)
{
	if (!valid(handle)) { return; }
	u64 const offset = allocator.elementOffset(srat::handle_index(handle.id));
	SRAT_ASSERT(offset < maxHandles);
	resourceAllocations[offset].~InternalResource();
#if SRAT_DEBUG()
	locationMap.erase(handle.id);
#endif
	allocator.free(handle.id);
}

template <typename Handle, typename InternalResource>
bool srat::HandlePool<Handle, InternalResource>::valid(
	Handle const & handle
) const {
	if (handle.id == 0) { return false; }
	if (!allocator.isHandleAlive(handle.id)) { return false; }
	SRAT_ASSERT(
		allocator.elementOffset(srat::handle_index(handle.id)) < maxHandles
	);
	return true;
}

template <typename Handle, typename InternalResource>
InternalResource * srat::HandlePool<Handle, InternalResource>::get(
	Handle const & handle
) {
	if (!valid(handle)) { return nullptr; }
	u32 const offset = allocator.elementOffset(srat::handle_index(handle.id));
	SRAT_ASSERT(offset < maxHandles);
	return &resourceAllocations[offset];
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
