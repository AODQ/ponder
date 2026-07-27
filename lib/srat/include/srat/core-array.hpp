#pragma once

#include <srat/core-types.hpp>

namespace srat {

template <typename T, size_t N>
struct array;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
static constexpr size_t skSliceDynamicExtent = ~0zu;
template <typename T, size_t N = skSliceDynamicExtent>
struct slice {
private:
	// -- data
	T * data { nullptr };
	size_t elemCount { N == skSliceDynamicExtent ? size_t(0) : N };
public:
	// -- constructors
	constexpr slice() = default;
	constexpr slice(T * const data, size_t const elemCount)
		: data(data), elemCount(elemCount)
	{
		if (N != skSliceDynamicExtent) {
			SRAT_ASSERT(elemCount == N);
		}
	}
	template <size_t M>
	constexpr slice(srat::array<T, M> const & arr)
		requires(N == skSliceDynamicExtent || N == M)
		: data(const_cast<T *>(arr.data)), elemCount(M) // NOLINT
	{
		if (N != skSliceDynamicExtent) {
			SRAT_ASSERT(elemCount == N);
		}
	}
	constexpr slice(T * const data) requires(N != skSliceDynamicExtent)
		: data(data), elemCount(N)
	{
		SRAT_ASSERT(elemCount != skSliceDynamicExtent);
	}
	constexpr slice(slice const &) = default;
	constexpr slice(slice &&) = default;
	constexpr slice & operator=(slice const &) = default;
	constexpr slice & operator=(slice &&) = default;
	constexpr ~slice() = default;

	// -- type aliases
	using value_type = T;
	using size_type = size_t;
	using difference_type = std::ptrdiff_t;
	using reference = T &;
	using const_reference = const T &;
	using pointer = T *;
	using const_pointer = const T *;
	using iterator = T *;
	using const_iterator = const T *;

	// -- element access
	[[nodiscard]] reference operator[](size_type const i) {
		SRAT_ASSERT(elemCount != skSliceDynamicExtent);
		SRAT_ASSERT(i < elemCount);
		return data[i];
	}

	[[nodiscard]] reference operator[](size_type const i) const {
		SRAT_ASSERT(elemCount != skSliceDynamicExtent);
		SRAT_ASSERT(i < elemCount);
		return data[i];
	}

	[[nodiscard]] reference front() { return operator[](0); }
	[[nodiscard]] reference front() const { return operator[](0); }
	[[nodiscard]] reference back() { return operator[](elemCount - 1); }
	[[nodiscard]] reference back() const
		{ return operator[](elemCount - 1); }
	[[nodiscard]] pointer ptr() {
		SRAT_ASSERT((data == nullptr) == (elemCount == 0));
		SRAT_ASSERT(elemCount != skSliceDynamicExtent);
		return data;
	}
	[[nodiscard]] const_pointer ptr() const { return data; }
	[[nodiscard]] size_type size() const {
		SRAT_ASSERT(elemCount != skSliceDynamicExtent);
		return elemCount;
	}

	// -- capacity
	[[nodiscard]] bool empty() const { return elemCount == 0; }

	// -- iterators
	[[nodiscard]] iterator begin() { return data; }
	[[nodiscard]] const_iterator begin() const { return data; }
	[[nodiscard]] iterator end() { return data + elemCount; }
	[[nodiscard]] const_iterator end() const { return data + elemCount; }
	[[nodiscard]] const_iterator cbegin() const { return data; }
	[[nodiscard]] const_iterator cend() const { return data + elemCount; }

	// -- convenience
	constexpr void splat(T const & value) {
		for (auto i = 0zu; i < elemCount; ++i) { data[i] = value; }
	}
	constexpr slice<T> subslice(
		size_t const offset,
		size_t const countOpt = skSliceDynamicExtent
	) const {
		SRAT_ASSERT(offset <= elemCount);
		Let count = size_t {
			countOpt == skSliceDynamicExtent ? (elemCount - offset) : countOpt
		};
		SRAT_ASSERT(offset + count <= elemCount);
		return slice<T> { data + offset, count };
	}
	template <size_t Count>
	constexpr slice<T, Count> as() const {
		SRAT_ASSERT(elemCount >= Count);
		return slice<T, Count> { data, Count };
	};
	template <typename U>
	constexpr slice<U> cast() const {
		SRAT_ASSERT(sizeof(T) * elemCount % sizeof(U) == 0);
		return slice<U> {
			reinterpret_cast<U *>(data),
			(sizeof(T) * elemCount) / sizeof(U)
		};
	}

	constexpr operator slice<T const, N>() const
		{ return slice<T const, N> { data, elemCount }; }
};
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

template <typename T, size_t N>
struct array {
	// -- data
	T data[N]; // NOLINT

	// -- type aliases
	using value_type = T;
	using size_type = size_t;
	using difference_type = std::ptrdiff_t;
	using reference = T &;
	using const_reference = const T &;
	using pointer = T *;
	using const_pointer = const T *;
	using iterator = T *;
	using const_iterator = const T *;

	// -- element access
	[[nodiscard]] constexpr reference operator[](size_type const i) {
		SRAT_ASSERT(i < N);
		return data[i];
	}

	[[nodiscard]] constexpr const_reference operator[](size_type const i) const {
		SRAT_ASSERT(i < N);
		return data[i];
	}

	[[nodiscard]] constexpr reference front() { return operator[](0); }
	[[nodiscard]] constexpr const_reference front() const
		{ return operator[](0); }
	[[nodiscard]] constexpr reference back() { return operator[](N - 1); }
	[[nodiscard]] constexpr const_reference back() const
		{ return operator[](N - 1); }
	[[nodiscard]] constexpr pointer ptr() { return data; }
	[[nodiscard]] constexpr const_pointer ptr() const { return data; }

	// -- capacity
	[[nodiscard]] static constexpr size_type size() { return N; }
	[[nodiscard]] static constexpr bool empty() { return N == 0; }

	// -- iterators
	[[nodiscard]] constexpr iterator begin() { return data; }
	[[nodiscard]] constexpr const_iterator begin() const { return data; }
	[[nodiscard]] constexpr iterator end() { return data + N; }
	[[nodiscard]] constexpr const_iterator end() const { return data + N; }
	[[nodiscard]] constexpr const_iterator cbegin() const { return data; }
	[[nodiscard]] constexpr const_iterator cend() const { return data + N; }

	// -- operations
	constexpr void splat(T const &value) {
		for (size_type i = 0; i < N; ++i) { data[i] = value; }
	}
	constexpr slice<T> subslice(
		size_t const offset,
		size_t const countOpt = skSliceDynamicExtent
	) const {
		SRAT_ASSERT(offset <= N);
		Let count = size_t {
			countOpt == skSliceDynamicExtent ? (N - offset) : countOpt
		};
		SRAT_ASSERT(offset + count <= N);
		return slice<T> { data + offset, count };
	}

	[[nodiscard]] constexpr operator slice<T, N>()
		{ return slice<T, N> { data }; }
	[[nodiscard]] constexpr operator slice<T, N>() const
		{ return slice<T, N> { const_cast<T *>(data), N }; } // NOLINT
};

template<typename T>
inline slice<u8 const> slice_as_bytes(T const & value) {
	return { reinterpret_cast<u8 const *>(&value), sizeof(T) };
}

} // namespace srat
