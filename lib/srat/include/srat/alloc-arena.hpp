#pragma once

#include <srat/core-types.hpp>
#include <srat/alloc-virtual-range.hpp>

#include <tuple>

// arena allocator for temporary memory allocations during rasterization

// -----------------------------------------------------------------------------
// -- arena allocator
// -----------------------------------------------------------------------------

namespace srat {

template <typename T>
struct AllocArena {
	static AllocArena<T> create(
		u32 const capacity,
		char const * const debugName = "ArenaAllocator"
	);

	// allocate N contiguous elements
	srat::slice<T> allocate(u32 const elementCount, u32 alignment = alignof(T));

	inline srat::slice<T> data_ptr() {
		if (allocator.allocatedCount() == 0) {
			return srat::slice<T>(nullptr, 0);
		}
		return srat::slice<T>(data, allocator.allocatedCount());
	}

	// avoid using below, prefer data_ptr if possible. The reason to use
	// data_ptr_raw is if you only want the base ptr, data_ptr can return null
	// if no allocations were made yet.
	inline T * data_ptr_raw() { return data; }

	void clear();
	[[nodiscard]] bool empty() const;
	[[nodiscard]] u32 capacity() const { return elemCapacity; }

	~AllocArena();

	AllocArena(AllocArena const &) = delete;
	AllocArena & operator=(AllocArena const &) = delete;
	AllocArena(AllocArena &&);
	AllocArena & operator=(AllocArena &&);
private:
	AllocArena(
		AllocVirtualRange && allocator, T * const data, u32 const capacity
	);

	AllocVirtualRange allocator;
	T * data;
	u32 elemCapacity;
};

}

// -----------------------------------------------------------------------------
// -- struct of array arena allocator
// -----------------------------------------------------------------------------

namespace srat {

template <typename ... Ts>
struct AllocArenaSoA {
	// create an arena allocator for each type, with given capacity
	static AllocArenaSoA<Ts ...> create(
		u32 const capacity,
		char const * const debugName = "SoAArenaAllocator"
	) {
		return AllocArenaSoA<Ts...>(
			std::make_tuple(AllocArena<Ts>::create(capacity, debugName) ...)
		);
	}
	// allocate N contiguous elements for each type, returns tuple of pointers
	[[nodiscard]]
	std::tuple<srat::slice<Ts> ...> allocate(u32 const elementCount) {
		return allocate_impl(elementCount, std::index_sequence_for<Ts ...>{});
	}

	[[nodiscard]] std::tuple<srat::slice<Ts> ...> data_ptr() {
		return data_ptr_impl(std::index_sequence_for<Ts ...>{});
	}

	[[nodiscard]] bool empty() const {
		return empty_impl(std::index_sequence_for<Ts ...>{});
	}

	[[nodiscard]] u32 capacity() const {
		return std::get<0>(allocators).capacity();
	}

	// clear all allocators
	void clear() {
		clear_impl(std::index_sequence_for<Ts ...>{});
	}

private:

	AllocArenaSoA(std::tuple<AllocArena<Ts> ...> && allocators) :
		allocators(std::move(allocators))
	{}

	template <usize ... Is>
	std::tuple<srat::slice<Ts> ...> allocate_impl(
		u32 const elementCount,
		std::index_sequence<Is ...>
	) {
		Let slices = std::make_tuple(
			std::get<Is>(allocators).allocate(elementCount) ...
		);
		SRAT_ASSERT(
			// if any allocation failed, they must all fail (or all succeed)
			((std::get<Is>(slices).ptr() == nullptr) || ... || true) &&
			((std::get<Is>(slices).ptr() != nullptr) || ... || true)
		);
		return slices;
	}

	template <usize ... Is>
	std::tuple<srat::slice<Ts> ...> data_ptr_impl(std::index_sequence<Is ...>) {
		return std::make_tuple(std::get<Is>(allocators).data_ptr() ...);
	}

	template <usize ... Is>
	bool empty_impl(std::index_sequence<Is ...>) const {
		return (std::get<Is>(allocators).empty() && ...);
	}

	template <usize ... Is>
	void clear_impl(
		std::index_sequence<Is ...>
	) {
		(std::get<Is>(allocators).clear(), ...);
	}


	std::tuple<AllocArena<Ts> ...> allocators;
};

}

// -----------------------------------------------------------------------------
// -- arena allocator implementation
// -----------------------------------------------------------------------------

#include <new> // sadly, for std::align_val_t

template <typename T>
srat::AllocArena<T>::AllocArena(
	AllocVirtualRange && allocator, T * const data, u32 const capacity
)
	: allocator(std::move(allocator))
	, data(data)
	, elemCapacity(capacity)
{}

template <typename T>
srat::AllocArena<T> srat::AllocArena<T>::create(
	u32 const capacity,
	char const * const debugName
) {
	return AllocArena<T>(
		AllocVirtualRange::create(AllocVirtualRangeCreateParams {
			.debugName = debugName,
			.elementCount = capacity,
			.maxBlockAllocations = 0, // not used for linear strategy
			.strategy = AllocVirtualRangeStrategy::Linear,
		}),
		static_cast<T *>(
			operator new[](sizeof(T) * capacity, std::align_val_t{alignof(T)})
		),
		capacity
	);
}

template <typename T>
srat::AllocArena<T>::AllocArena(AllocArena && o)
	: allocator(std::move(o.allocator))
	, data(o.data)
	, elemCapacity(o.elemCapacity)
{
	o.data = nullptr;
	o.elemCapacity = 0;
}

template <typename T>
srat::AllocArena<T> & srat::AllocArena<T>::operator=(
	AllocArena && o
) {
	if (this != &o) {
		allocator = std::move(o.allocator);
		// delete existing data before taking ownership of new data
		if (data != nullptr) {
			operator delete[](data, std::align_val_t{alignof(T)});
		}
		data = o.data;
		elemCapacity = o.elemCapacity;
		o.data = nullptr;
		o.elemCapacity = 0;
	}
	return *this;
}

template <typename T>
srat::AllocArena<T>::AllocArena::~AllocArena() {
	if (data != nullptr) {
		operator delete[](data, std::align_val_t{alignof(T)});
	}
	data = nullptr;
}

template <typename T>
srat::slice<T> srat::AllocArena<T>::allocate(
	u32 const elementCount,
	u32 const alignment
) {
	Let block = (
		allocator.allocate({
			.elementCount = elementCount,
			.elementAlignment = alignment,
		})
	);
	if (block.elementCount == 0) {
		return {nullptr, 0}; // allocation failed
	}
	return {
		data + block.elementOffset,
		block.elementCount,
	};
}

template <typename T>
void srat::AllocArena<T>::clear() {
	allocator.clear();
}

template <typename T>
bool srat::AllocArena<T>::empty() const {
	return allocator.empty();
}
