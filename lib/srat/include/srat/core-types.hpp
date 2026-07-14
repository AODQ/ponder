#pragma once

// common types + includes for srat

#include <cstdint>
#include <utility> // for std::move
#include <cstdio> // TODO move to logging

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8 = uint8_t;
using i64 = int64_t;
using i32 = int32_t;
using i16 = int16_t;
using i8 = int8_t;
using f32 = float;
using f64 = double;
using usize = std::size_t;

static constexpr auto skEpsilon = 0.0001f;

// common types for the rasterizer binning stage

struct i32v2;
struct f32v4;

namespace srat {

using triangle_index_t = u32;
using triangle_count_t = u32;
using triangle_position_t = i32v2;
using triangle_depth_t = float;
using triangle_perspective_w_t = float;
using triangle_color_t = f32v4;

template <typename T>
inline T alignUp(T value, T alignment)
{
	return (value + (alignment - 1)) & ~(alignment - 1);
}

} // srat

// -----------------------------------------------------------------------------
// color

struct ColorRgba8 {
	u8 r, g, b, a;
};

// -----------------------------------------------------------------------------
// logging utility

#include <srat/core-config.hpp>

#include <cstdio>
#include <cstdlib>
#define SRAT_ASSERT_ALWAYS(expr) { \
	if (!(expr)) { \
		fprintf(stderr, "Assertion failed: %s\nFile: %s\nLine: %d\n", \
			#expr, __FILE__, __LINE__); \
		std::abort(); \
	} \
}
// #if SRAT_DEBUG()
#define SRAT_ASSERT(expr) SRAT_ASSERT_ALWAYS(expr)
// #else
// #define SRAT_ASSERT(expr)
// #endif
