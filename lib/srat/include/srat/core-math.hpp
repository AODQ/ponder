#pragma once

#include <srat/core-types.hpp>
#include <srat/core-array.hpp>

#include <cmath>
#include <cstring>
#include <immintrin.h>

// -----------------------------------------------------------------------------
// -- T math
// -----------------------------------------------------------------------------

// only handles positive values correctly
template <typename T>
[[nodiscard]] inline T T_roundf_positive(f32 const v) {
	SRAT_ASSERT(v >= 0.0f);
	return static_cast<T>(v + 0.5f);
}

// -----------------------------------------------------------------------------
// -- i32 math
// -----------------------------------------------------------------------------

inline i32 i32_min(i32 const a, i32 const b) { return (a < b) ? a : b; }
inline i32 i32_max(i32 const a, i32 const b) { return (a > b) ? a : b; }
inline i32 i32_abs(i32 const a) { return (a < 0) ? -a : a; }
inline i32 i32_clamp(i32 const v, i32 const min, i32 const max) {
	return i32_min(i32_max(v, min), max);
}

// -----------------------------------------------------------------------------
// -- u32 math
// -----------------------------------------------------------------------------

inline u32 u32_min(u32 const a, u32 const b) { return (a < b) ? a : b; }
inline u32 u32_max(u32 const a, u32 const b) { return (a > b) ? a : b; }
inline u32 u32_clamp(u32 const v, u32 const min, u32 const max) {
	return u32_min(u32_max(v, min), max);
}

// -----------------------------------------------------------------------------
// -- u32v2
// -----------------------------------------------------------------------------

struct u32v2 {
	u32 x, y;

	u32v2 operator+(u32v2 const s) const { return { x+s.x, y+s.y, }; }
	u32v2 operator-(u32v2 const s) const { return { x-s.x, y-s.y, }; }
	u32v2 operator*(u32v2 const s) const { return { x*s.x, y*s.y, }; }
	u32v2 operator/(u32v2 const s) const { return { x/s.x, y/s.y, }; }

	bool operator==(u32v2 const & s) const = default;
};

inline u32v2 u32v2_min(u32v2 const a, u32v2 const b) {
	return {
		u32_min(a.x, b.x),
		u32_min(a.y, b.y),
	};
}

// -----------------------------------------------------------------------------
// -- i32v2
// -----------------------------------------------------------------------------

struct i32v2 {
	i32 x, y;

	i32v2 operator *(i32v2 const s) const { return { x*s.x, y*s.y, }; }
	i32v2 operator /(i32v2 const s) const { return { x/s.x, y/s.y, }; }
	i32v2 operator -(i32v2 const s) const { return { x-s.x, y-s.y, }; }
	i32v2 operator +(i32v2 const s) const { return { x+s.x, y+s.y, }; }

	bool operator==(i32v2 const & s) const = default;
};

inline i32v2 i32v2_clamp(i32v2 const v, i32v2 const min, i32v2 const max) {
	return {
		i32_clamp(v.x, min.x, max.x),
		i32_clamp(v.y, min.y, max.y),
	};
}
inline i32v2 i32v2_min(i32v2 const a, i32v2 const b) {
	return {
		i32_min(a.x, b.x),
		i32_min(a.y, b.y),
	};
}

// -----------------------------------------------------------------------------
// -- u32v3
// -----------------------------------------------------------------------------

struct u32v3 {
	u32 x, y, z;

	u32v3 operator+(u32v3 const s) const { return { x+s.x, y+s.y, z+s.z}; }
	u32v3 operator-(u32v3 const s) const { return { x-s.x, y-s.y, z-s.z}; }
	u32v3 operator*(u32v3 const s) const { return { x*s.x, y*s.y, z*s.z}; }
	u32v3 operator/(u32v3 const s) const { return { x/s.x, y/s.y, z/s.z}; }

	bool operator==(u32v3 const & s) const = default;
};

inline u32v3 u32v3_min(u32v3 const a, u32v3 const b) {
	return {
		u32_min(a.x, b.x),
		u32_min(a.y, b.y),
		u32_min(a.z, b.z),
	};
}

// -----------------------------------------------------------------------------
// -- f32v2
// -----------------------------------------------------------------------------

struct f32v2 {
	f32 x, y;

	[[nodiscard]] constexpr inline f32v2 operator /(f32v2 const s) const
		{ return { x/s.x, y/s.y, }; }
	[[nodiscard]] constexpr inline f32v2 operator /(f32 const s) const
		{ return { x/s, y/s, }; }
	[[nodiscard]] constexpr inline f32v2 operator -(f32v2 const s) const
		{ return { x-s.x, y-s.y, }; }
	[[nodiscard]] constexpr inline f32v2 operator +(f32v2 const s) const
		{ return { x+s.x, y+s.y, }; }
	[[nodiscard]] constexpr inline f32v2 operator *(f32v2 const s) const
		{ return { x*s.x, y*s.y, }; }
	[[nodiscard]] constexpr inline f32v2 operator *(f32 const s) const
		{ return { x*s, y*s, }; }

	constexpr inline f32v2 operator+=(f32v2 const s) {
		x += s.x; y += s.y;
		return *this;
	}
};

// -----------------------------------------------------------------------------
// -- f32v3
// -----------------------------------------------------------------------------

struct f32v3 {
	f32 x, y, z;

	[[nodiscard]] f32v2 xy() const { return { x, y }; }

	f32v3 operator /(f32v3 const s) const { return { x/s.x, y/s.y, z/s.z, }; }
	f32v3 operator /(f32 const s) const { return { x/s, y/s, z/s, }; }
	f32v3 operator -(f32v3 const s) const { return { x-s.x, y-s.y, z-s.z, }; }
	f32v3 operator +(f32v3 const s) const { return { x+s.x, y+s.y, z+s.z, }; }
	f32v3 operator *(f32v3 const s) const { return { x*s.x, y*s.y, z*s.z, }; }
	f32v3 operator *(f32 const s) const { return { x*s, y*s, z*s, }; }
};

[[nodiscard]] constexpr inline f32v3 f32v3_normalize(f32v3 const v) {
	f32 const lenSq = v.x*v.x + v.y*v.y + v.z*v.z;
	if (lenSq > 0.0f) {
		f32 const invLen = 1.0f / std::sqrtf(lenSq);
		return { v.x*invLen, v.y*invLen, v.z*invLen };
	}
	else {
		return { 0.0f, 0.0f, 0.0f };
	}
}

[[nodiscard]] constexpr inline f32 f32v3_dot(f32v3 const a, f32v3 const b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}


inline f32 f32v3_length(f32v3 const v) {
	return std::sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

[[nodiscard]] constexpr inline f32v3 f32v3_cross(f32v3 const a, f32v3 const b) {
	return {
		a.y*b.z - a.z*b.y,
		a.z*b.x - a.x*b.z,
		a.x*b.y - a.y*b.x
	};
}

// -----------------------------------------------------------------------------
// -- f32v4
// -----------------------------------------------------------------------------

struct f32v4 {
	f32 x {0.0f}, y {0.0f}, z {0.0f}, w {0.0f};

	constexpr f32v4() = default;
	constexpr f32v4(f32v3 const & v3, f32 w) : x(v3.x), y(v3.y), z(v3.z), w(w) {}
	constexpr f32v4(f32 const x, f32 const y, f32 const z, f32 const w) //NOLINT
		: x(x), y(y), z(z), w(w) {}

	[[nodiscard]] constexpr f32v2 xy() const { return { x, y }; }
	[[nodiscard]] constexpr f32v3 xyz() const { return { x, y, z }; }

	[[nodiscard]] constexpr f32v4 operator /(f32v4 const s) const {
		return { x/s.x, y/s.y, z/s.z, w/s.w };
	}
	[[nodiscard]] constexpr f32v4 operator /(f32 const s) const {
		return { x/s, y/s, z/s, w/s };
	}
	[[nodiscard]] constexpr f32v4 operator -(f32v4 const s) const {
		return { x-s.x, y-s.y, z-s.z, w-s.w };
	}
	[[nodiscard]] constexpr f32v4 operator +(f32v4 const s) const {
		return { x+s.x, y+s.y, z+s.z, w+s.w };
	}
	[[nodiscard]] constexpr f32v4 operator *(f32v4 const s) const {
		return { x*s.x, y*s.y, z*s.z, w*s.w };
	}
	[[nodiscard]] constexpr f32v4 operator *(f32 const s) const {
		return { x*s, y*s, z*s, w*s };
	}
	constexpr f32v4 operator+=(f32v4 const s) {
		x += s.x; y += s.y; z += s.z; w += s.w;
		return *this;
	}
};

// -----------------------------------------------------------------------------
// -- i32x8/u32x8
// -----------------------------------------------------------------------------

struct i32x8 {
	__m256i v;
	[[nodiscard]] inline i32x8 operator&(i32x8 const & o) const {
		return { _mm256_and_si256(v, o.v) };
	}
	[[nodiscard]] inline i32x8 operator|(i32x8 const & o) const {
		return { _mm256_or_si256(v, o.v) };
	}
	[[nodiscard]] inline i32x8 operator^(i32x8 const & o) const {
		return { _mm256_xor_si256(v, o.v) };
	}
	[[nodiscard]] inline i32x8 operator~() const {
		return { _mm256_xor_si256(v, _mm256_set1_epi32(-1)) };
	}
	[[nodiscard]] inline i32x8 operator+(i32x8 const & o) const {
		return { _mm256_add_epi32(v, o.v) };
	}
	[[nodiscard]] inline i32x8 operator-(i32x8 const & o) const {
		return { _mm256_sub_epi32(v, o.v) };
	}

	[[nodiscard]] inline i32x8 operator*(i32x8 const & o) const {
		return { _mm256_mullo_epi32(v, o.v) };
	}
};

struct u32x8 {
	__m256i v;
	u32x8 operator&(u32x8 const & o) const {
		return { _mm256_and_si256(v, o.v) };
	}
	u32x8 operator|(u32x8 const & o) const {
		return { _mm256_or_si256(v, o.v) };
	}
	u32x8 operator^(u32x8 const & o) const {
		return { _mm256_xor_si256(v, o.v) };
	}
	u32x8 operator~() const {
		return { _mm256_xor_si256(v, _mm256_set1_epi32(-1)) };
	}

};

inline i32x8 i32x8_load(srat::array<i32, 8> const & in) {
	return { _mm256_loadu_si256(reinterpret_cast<__m256i const *>(in.ptr())) };
}
inline u32x8 u32x8_load(srat::array<u32, 8> const & in) {
	return { _mm256_loadu_si256(reinterpret_cast<__m256i const *>(in.ptr())) };
}
inline i32 i32x8_lane0(i32x8 const & v) { return _mm256_extract_epi32(v.v, 0); }
inline u32 u32x8_lane0(u32x8 const & v) { return _mm256_extract_epi32(v.v, 0); }
inline void i32x8_store(i32x8 const & v, srat::array<i32, 8> & out) {
	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out.ptr()), v.v);
}
inline void u32x8_store(u32x8 const & v, srat::array<u32, 8> & out) {
	_mm256_storeu_si256(reinterpret_cast<__m256i *>(out.ptr()), v.v);
}
inline i32x8 i32x8_splat(i32 i) { return { _mm256_set1_epi32(i) }; }
inline u32x8 u32x8_splat(u32 i) {
	i32 iSigned = *reinterpret_cast<i32 const *>(&i);
	return { _mm256_set1_epi32(iSigned) };
}
inline i32x8 i32x8_zero() { return i32x8_splat(0); }
inline u32x8 u32x8_zero() { return u32x8_splat(0); }

inline u32 u32x8_ballot(u32x8 const & v) {
	return (u32)_mm256_movemask_ps(_mm256_castsi256_ps(v.v));
}
inline bool u32x8_any(u32x8 const & v) { return u32x8_ballot(v) != 0; }
inline bool u32x8_all(u32x8 const & v) { return u32x8_ballot(v) == 0xFF; }
[[nodiscard]] inline bool u32x8_mask_sign_bit(u32x8 const & v)
	{ return _mm256_movemask_epi8(v.v) != 0; }

// -----------------------------------------------------------------------------
// -- f32x8
// -----------------------------------------------------------------------------

struct f32x8 {
	__m256 v;

	[[nodiscard]] f32 lane(size_t lane) const {
		//NOLINTBEGIN
		#define extract_low(x) \
			_mm256_cvtss_f32(_mm256_permute_ps(v, _MM_SHUFFLE(x,x,x,x))) \

		#define extract_high(x) \
			_mm_cvtss_f32( \
				_mm_permute_ps( \
					_mm256_extractf128_ps(v, 1), \
					_MM_SHUFFLE(x,x,x,x) \
				) \
			) \
		//NOLINTEND

		switch (lane) {
			case 0: return _mm256_cvtss_f32(v);
			case 1: return extract_low(1);
			case 2: return extract_low(2);
			case 3: return extract_low(3);
			case 4: return _mm_cvtss_f32(_mm256_extractf128_ps(v, 1));
			case 5: return extract_high(1);
			case 6: return extract_high(2);
			case 7: return extract_high(3);
		}
		return 0.0f; // silence warning
	}

	[[nodiscard]] inline f32x8 reciprocal() const
		{ return { _mm256_rcp_ps(v) }; }
	[[nodiscard]] inline f32x8 reciprocal_newton_raphson() const {
		// one iteration of Newton-Raphson refinement for better accuracy
		f32x8 rcp = reciprocal();
		return rcp * (f32x8 { _mm256_set1_ps(2.0f) } - (*this * rcp));
	}

	[[nodiscard]] inline f32x8 operator+(f32x8 const & o) const
		{ return { _mm256_add_ps(v, o.v) }; }
	[[nodiscard]] inline f32x8 operator-(f32x8 const & o) const
		{ return { _mm256_sub_ps(v, o.v) }; }
	[[nodiscard]] inline f32x8 operator*(f32x8 const & o) const
		{ return { _mm256_mul_ps(v, o.v) }; }
	[[nodiscard]] inline f32x8 operator/(f32x8 const & o) const
		{ return { _mm256_div_ps(v, o.v) }; }


	[[nodiscard]] inline f32x8 operator-() const
		{ return { _mm256_sub_ps(_mm256_setzero_ps(), v) }; }

	[[nodiscard]] inline u32x8 operator<(f32x8 const & o) const
		{ return { _mm256_castps_si256(_mm256_cmp_ps(v, o.v, _CMP_LT_OQ)) }; }
	[[nodiscard]] inline u32x8 operator>(f32x8 const & o) const
		{ return { _mm256_castps_si256(_mm256_cmp_ps(v, o.v, _CMP_GT_OQ)) }; }
	[[nodiscard]] inline u32x8 operator<=(f32x8 const & o) const
		{ return { _mm256_castps_si256(_mm256_cmp_ps(v, o.v, _CMP_LE_OQ)) }; }
	[[nodiscard]] inline u32x8 operator>=(f32x8 const & o) const
		{ return { _mm256_castps_si256(_mm256_cmp_ps(v, o.v, _CMP_GE_OQ)) }; }
};

inline f32x8 f32x8_from_i32x8(i32x8 const & v) {
	return { _mm256_cvtepi32_ps(v.v) };
}

inline u32x8 u32x8_from_f32x8(f32x8 const & v) {
	return { _mm256_cvtps_epi32(v.v) };
}

inline f32x8 f32x8_load(srat::slice<f32, 8> const & in)
	{ return { _mm256_loadu_ps(in.ptr()) }; }
inline f32 f32x8_lane0(f32x8 const & v) { return _mm256_cvtss_f32(v.v); }
inline void f32x8_store(f32x8 const & v, srat::slice<f32, 8> out)
	{ _mm256_storeu_ps(out.ptr(), v.v); }
[[nodiscard]] constexpr inline f32x8 f32x8_splat(f32 f)
	{ return { _mm256_set1_ps(f) }; }
constexpr inline f32x8 f32x8_zero() { return f32x8_splat(0.0f); }

// Fused multiply-add: a * b + c
inline f32x8 f32x8_fmadd(f32x8 const & a, f32x8 const & b, f32x8 const & c) {
	return { _mm256_fmadd_ps(a.v, b.v, c.v) };
}
// blend based on select: if mask is true, select a else select b
inline f32x8 f32x8_select(u32x8 const & mask, f32x8 const & a, f32x8 const & b) {
	return { _mm256_blendv_ps(a.v, b.v, _mm256_castsi256_ps(mask.v)) };
}

// min/max
inline f32x8 f32x8_min(f32x8 const & a, f32x8 const & b)
	{ return { _mm256_min_ps(a.v, b.v) }; }
inline f32x8 f32x8_max(f32x8 const & a, f32x8 const & b)
	{ return { _mm256_max_ps(a.v, b.v) }; }
inline f32x8 f32x8_clamp(f32x8 const & v, f32x8 const & min, f32x8 const & max) {
	return f32x8_min(f32x8_max(v, min), max);
}

[[nodiscard]] constexpr inline f32v3 f32v3_min(
	f32v3 const & a, f32v3 const & b
) {
	return {
		std::fminf(a.x, b.x),
		std::fminf(a.y, b.y),
		std::fminf(a.z, b.z),
	};
}
[[nodiscard]] constexpr inline f32v3 f32v3_max(
	f32v3 const & a, f32v3 const & b
) {
	return {
		std::fmaxf(a.x, b.x),
		std::fmaxf(a.y, b.y),
		std::fmaxf(a.z, b.z),
	};
}

// sqrt
inline f32x8 f32x8_sqrt(f32x8 const & v) { return { _mm256_sqrt_ps(v.v) }; }
inline f32x8 f32x8_rsqrt(f32x8 const & v) { return { _mm256_rsqrt_ps(v.v) }; }

// -----------------------------------------------------------------------------
// -- i32v2x8
// -----------------------------------------------------------------------------

struct i32v2x8 {
	union {
		srat::array<i32x8, 2> v;
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wpedantic"
		struct { i32x8 x, y; };
		#pragma GCC diagnostic pop
	};

	i32v2x8() = default;
	i32v2x8(i32x8 const & x, i32x8 const & y) : v{ x, y } {}

	i32v2x8 operator+(i32v2x8 const & o) const {
		return {
			v[0] + o.v[0],
			v[1] + o.v[1],
		};
	}
	i32v2x8 operator-(i32v2x8 const & o) const {
		return {
			v[0] - o.v[0],
			v[1] - o.v[1],
		};
	}
};

struct f32v2x8;

i32v2x8 i32v2x8_from_f32v2x8(f32v2x8 const & v);

// -----------------------------------------------------------------------------
// -- f32v2x8
// -----------------------------------------------------------------------------

struct f32v2x8 {
	union {
		srat::array<f32x8, 2> v;
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wpedantic"
		struct { f32x8 x, y; };
		#pragma GCC diagnostic pop
	};

	f32v2x8() = default;
	f32v2x8(f32x8 const & x, f32x8 const & y) : v{ x, y } {}

	f32v2x8 operator+(f32v2x8 const & o) const {
		return {
			v[0] + o.v[0],
			v[1] + o.v[1],
		};
	}
	f32v2x8 operator-(f32v2x8 const & o) const {
		return {
			v[0] - o.v[0],
			v[1] - o.v[1],
		};
	}
	f32v2x8 operator*(f32v2x8 const & o) const {
		return {
			v[0] * o.v[0],
			v[1] * o.v[1],
		};
	}
	f32v2x8 operator/(f32v2x8 const & o) const {
		return {
			v[0] / o.v[0],
			v[1] / o.v[1],
		};
	}

	f32v2x8 operator*(f32x8 const & s) const {
		return { v[0] * s, v[1] * s, };
	}
};

inline f32v2x8 f32v2x8_load(srat::slice<f32, 16> const & in) {
	return f32v2x8 {
		f32x8_load(in.as<8>()),
		f32x8_load(in.subslice(8).as<8>()),
	};
}
inline f32v2x8 f32v2x8_splat(srat::slice<f32, 2> in) {
	return {
		f32x8_splat(in[0]),
		f32x8_splat(in[1]),
	};
}
inline f32v2x8 f32v2x8_splat(f32v2 const & v) {
	return {
		f32x8_splat(v.x),
		f32x8_splat(v.y),
	};
}

inline f32v2x8 f32v2x8_fract(f32v2x8 const & v) {
	auto const floorX = _mm256_floor_ps(v.v[0].v);
	auto const floorY = _mm256_floor_ps(v.v[1].v);
	return {
		{ _mm256_sub_ps(v.v[0].v, floorX) },
		{ _mm256_sub_ps(v.v[1].v, floorY) },
	};
}

inline f32v2x8 f32v2x8_splat(f32 const x, f32 const y) {
	return {
		f32x8_splat(x),
		f32x8_splat(y),
	};
}

// -----------------------------------------------------------------------------
// -- f32x8x3
// -----------------------------------------------------------------------------

struct f32v3x8 {
	srat::array<f32x8, 3> v;

	f32v3x8() = default;
	f32v3x8(f32x8 const & x, f32x8 const & y, f32x8 const & z) : v{ x, y, z }
	{
	}

	f32v3x8 operator+(f32v3x8 const & o) const {
		return {
			v[0] + o.v[0],
			v[1] + o.v[1],
			v[2] + o.v[2],
		};
	}

	f32v3x8 operator-(f32v3x8 const & o) const {
		return {
			v[0] - o.v[0],
			v[1] - o.v[1],
			v[2] - o.v[2],
		};
	}

	f32v3x8 operator*(f32v3x8 const & o) const {
		return {
			v[0] * o.v[0],
			v[1] * o.v[1],
			v[2] * o.v[2],
		};
	}
	f32v3x8 operator*(f32x8 const & s) const {
		return {
			v[0] * s,
			v[1] * s,
			v[2] * s,
		};
	}

	f32v3x8 operator/(f32v3x8 const & o) const {
		return {
			v[0] / o.v[0],
			v[1] / o.v[1],
			v[2] / o.v[2],
		};
	}
};

// -----------------------------------------------------------------------------
// -- f32v4x8
// -----------------------------------------------------------------------------

struct f32v4x8 {
	union {
		srat::array<f32x8, 4> v;
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wpedantic"
		struct { f32x8 x, y, z, w; };
		#pragma GCC diagnostic pop
	};

	f32v4x8() = default;
	f32v4x8(f32x8 const & x, f32x8 const & y, f32x8 const & z, f32x8 const & w)
		: v{ x, y, z, w }
	{
	}

	f32v4x8(f32v3x8 const & v3, f32x8 const & w)
		: v{ v3.v[0], v3.v[1], v3.v[2], w }
	{
	}

	[[nodiscard]] inline f32v3x8 xyz() const {
		return { v[0], v[1], v[2] };
	}

	f32v4x8 operator+(f32v4x8 const & o) const {
		return {
			v[0] + o.v[0],
			v[1] + o.v[1],
			v[2] + o.v[2],
			v[3] + o.v[3],
		};
	}

	f32v4x8 operator-(f32v4x8 const & o) const {
		return {
			v[0] - o.v[0],
			v[1] - o.v[1],
			v[2] - o.v[2],
			v[3] - o.v[3],
		};
	}

	f32v4x8 operator*(f32v4x8 const & o) const {
		return {
			v[0] * o.v[0],
			v[1] * o.v[1],
			v[2] * o.v[2],
			v[3] * o.v[3],
		};
	}

	f32v4x8 operator/(f32v4x8 const & o) const {
		return {
			v[0] / o.v[0],
			v[1] / o.v[1],
			v[2] / o.v[2],
			v[3] / o.v[3],
		};
	}

	f32v4x8 operator*(f32x8 const & s) const {
		return { v[0] * s, v[1] * s, v[2] * s, v[3] * s, };
	}
};

inline f32v4x8 f32v4x8_load(srat::slice<f32, 32> const & in) {
	return f32v4x8 {
		f32x8_load(in.as<8>()),
		f32x8_load(in.subslice(8).as<8>()),
		f32x8_load(in.subslice(16).as<8>()),
		f32x8_load(in.subslice(24).as<8>()),
	};
}

inline void f32v4x8_store(f32v4x8 const & v, srat::slice<f32, 32> const & out) {
	f32x8_store(v.v[0], out.as<8>());
	f32x8_store(v.v[1], out.subslice(8).as<8>());
	f32x8_store(v.v[2], out.subslice(16).as<8>());
	f32x8_store(v.v[3], out.subslice(24).as<8>());
}

inline f32v3x8 f32v3x8_splat(srat::slice<f32, 3> in) {
	return {
		f32x8_splat(in[0]),
		f32x8_splat(in[1]),
		f32x8_splat(in[2]),
	};
}

inline f32v3x8 f32v3x8_splat(f32 const x, f32 const y, f32 const z) {
	return {
		f32x8_splat(x),
		f32x8_splat(y),
		f32x8_splat(z),
	};
}

inline f32v3x8 f32v3x8_splat(f32v3 const & v) {
	return {
		f32x8_splat(v.x),
		f32x8_splat(v.y),
		f32x8_splat(v.z),
	};
}

inline f32v4x8 f32v4x8_splat(srat::slice<f32, 4> in) {
	return {
		f32x8_splat(in[0]),
		f32x8_splat(in[1]),
		f32x8_splat(in[2]),
		f32x8_splat(in[3]),
	};
}

inline f32v4x8 f32v4x8_splat(f32v4 const & v) {
	return {
		f32x8_splat(v.x),
		f32x8_splat(v.y),
		f32x8_splat(v.z),
		f32x8_splat(v.w),
	};
}

inline f32v4x8 f32v4x8_splat(
	f32 const x, f32 const y, f32 const z, f32 const w
) {
	return {
		f32x8_splat(x),
		f32x8_splat(y),
		f32x8_splat(z),
		f32x8_splat(w),
	};
}

inline f32v3x8 f32v3x8_zero() {
	static srat::array<f32, 3> zero = { 0.0f, 0.0f, 0.0f, };
	return f32v3x8_splat(zero);
}

inline f32v4x8 f32v4x8_zero() {
	static srat::array<f32, 4> zero = { 0.0f, 0.0f, 0.0f, 0.0f };
	return f32v4x8_splat(zero);
}

inline f32v4x8 f32v4x8_mix(
	f32v4x8 const & a, f32v4x8 const & b, f32v4x8 const & t
) {
	return {
		f32x8_fmadd(t.v[0], b.v[0] - a.v[0], a.v[0]),
		f32x8_fmadd(t.v[1], b.v[1] - a.v[1], a.v[1]),
		f32x8_fmadd(t.v[2], b.v[2] - a.v[2], a.v[2]),
		f32x8_fmadd(t.v[3], b.v[3] - a.v[3], a.v[3]),
	};
}

inline f32v4x8 f32v4x8_mix(
	f32v4x8 const & a, f32v4x8 const & b, f32x8 const & t
) {
	return {
		f32x8_fmadd(t, b.v[0] - a.v[0], a.v[0]),
		f32x8_fmadd(t, b.v[1] - a.v[1], a.v[1]),
		f32x8_fmadd(t, b.v[2] - a.v[2], a.v[2]),
		f32x8_fmadd(t, b.v[3] - a.v[3], a.v[3]),
	};
}

inline f32x8 f32v3x8_dot(f32v3x8 const & a, f32v3x8 const & b) {
	return {
		f32x8_fmadd(
			a.v[0], b.v[0],
			f32x8_fmadd(a.v[1], b.v[1], (a.v[2] * b.v[2]))
		)
	};
}

inline f32v3x8 f32v3x8_clamp(f32v3x8 const & v, f32v3x8 const & min, f32v3x8 const & max) {
	return {
		f32x8_clamp(v.v[0], min.v[0], max.v[0]),
		f32x8_clamp(v.v[1], min.v[1], max.v[1]),
		f32x8_clamp(v.v[2], min.v[2], max.v[2]),
	};
}

inline f32v3x8 f32v3x8_normalize(f32v3x8 const & v) {
	f32x8 const lengthSq = f32v3x8_dot(v, v);
	SRAT_ASSERT(lengthSq.v[0] > 0.0f);
	f32x8 const invLength = f32x8_rsqrt(lengthSq);
	return {
		v.v[0] * invLength,
		v.v[1] * invLength,
		v.v[2] * invLength
	};
}

inline f32x8 f32v4x8_dot(f32v4x8 const & a, f32v4x8 const & b) {
	return {
		f32x8_fmadd(
			a.v[0], b.v[0],
			f32x8_fmadd(
				a.v[1], b.v[1],
				f32x8_fmadd(a.v[2], b.v[2], (a.v[3] * b.v[3]))
			)
		)
	};
}

inline f32v4x8 f32v4x8_cross(f32v4x8 const & a, f32v4x8 const & b) {
	return {
		f32x8_fmadd(a.v[1], b.v[2], -(a.v[2] * b.v[1])),
		f32x8_fmadd(a.v[2], b.v[0], -(a.v[0] * b.v[2])),
		f32x8_fmadd(a.v[0], b.v[1], -(a.v[1] * b.v[0])),
		f32x8_zero(),
	};
}

inline f32v4x8 f32v4x8_normalize(f32v4x8 const & v) {
	f32x8 const lengthSq = f32v4x8_dot(v, v);
	SRAT_ASSERT(lengthSq.v[0] > 0.0f);
	f32x8 const invLength = f32x8_rsqrt(lengthSq);
	return {
		v.v[0] * invLength,
		v.v[1] * invLength,
		v.v[2] * invLength,
		f32x8_zero(),
	};
}

// -----------------------------------------------------------------------------
// -- f32m44
// -----------------------------------------------------------------------------

// column major matrix
struct f32m44 {
	srat::array<f32, 16> m;

	inline f32m44 operator*(f32m44 const & o) const {
		f32m44 result {};
		for (i32 col = 0; col < 4; ++col) {
			for (i32 row = 0; row < 4; ++row) {
				result.m[col*4 + row] = (
					  m[0*4 + row] * o.m[col*4 + 0]
					+ m[1*4 + row] * o.m[col*4 + 1]
					+ m[2*4 + row] * o.m[col*4 + 2]
					+ m[3*4 + row] * o.m[col*4 + 3]
				);
			}
		}
		return result;
	}

	inline f32v4 operator*(f32v4 const & v) const {
		return {
			m[0] * v.x + m[4] * v.y + m[8]  * v.z + m[12] * v.w,
			m[1] * v.x + m[5] * v.y + m[9]  * v.z + m[13] * v.w,
			m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
			m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w,
		};
	}
};

inline f32m44 f32m44_identity() {
	return {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f,
	};
};

inline f32m44 f32m44_translate(f32 x, f32 y, f32 z) {
	return {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		x,   y,   z,   1.0f,
	};
};

inline f32m44 f32m44_scale(f32 const x, f32 const y, f32 const z) {
	return {
		x,   0.0f, 0.0f, 0.0f,
		0.0f, y,   0.0f, 0.0f,
		0.0f, 0.0f, z,   0.0f,
		0.0f, 0.0f, 0.0f, 1.0f,
	};
};

inline f32m44 f32m44_rotate_z(f32 const angleRadians) {
	f32 const c = cosf(angleRadians);
	f32 const s = sinf(angleRadians);
	return {
		c,   s,   0.0f, 0.0f,
		-s,  c,   0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f,
	};
};

inline f32m44 f32m44_rotate_x(f32 const angleRadians) {
	f32 const c = cosf(angleRadians);
	f32 const s = sinf(angleRadians);
	return {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, c,   s,   0.0f,
		0.0f, -s,  c,   0.0f,
		0.0f, 0.0f, 0.0f, 1.0f,
	};
};

inline f32m44 f32m44_rotate_y(f32 const angleRadians) {
	f32 const c = cosf(angleRadians);
	f32 const s = sinf(angleRadians);
	return {
		c,   0.0f, -s,  0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		s,   0.0f, c,   0.0f,
		0.0f, 0.0f, 0.0f, 1.0f,
	};
};

inline f32m44 f32m44_perspective(
	f32 const fovY, f32 const aspect, f32 const zNear, f32 const zFar
) {
	f32 const f = 1.0f / tanf(fovY * 0.5f);
	f32 const d = zFar - zNear;
	return {
		f / aspect, 0.0f,  0.0f,                      0.0f,
		0.0f,       -f,    0.0f,                       0.0f,
		0.0f,       0.0f, -(zFar + zNear) / d,        -1.0f,
		0.0f,       0.0f, -(2.0f * zFar * zNear) / d,  0.0f,
	};
}

[[nodiscard]] inline constexpr f32m44 f32m44_lookat(
	f32v3 const & eye, f32v3 const & target, f32v3 const & up
) {
	f32v3 const z = f32v3_normalize(eye - target);
	f32v3 const x = f32v3_normalize(f32v3_cross(up, z));
	f32v3 const y = f32v3_cross(z, x);

	f32m44 m = f32m44_identity();

	m.m[0] = x.x; m.m[1] = y.x; m.m[2]  = z.x; m.m[3]  = 0.0f;
	m.m[4] = x.y; m.m[5] = y.y; m.m[6]  = z.y; m.m[7]  = 0.0f;
	m.m[8] = x.z; m.m[9] = y.z; m.m[10] = z.z; m.m[11] = 0.0f;

	m.m[12] = -f32v3_dot(x, eye);
	m.m[13] = -f32v3_dot(y, eye);
	m.m[14] = -f32v3_dot(z, eye);
	m.m[15] = 1.0f;

	return m;
}

// -----------------------------------------------------------------------------
// -- f32m44x8
// -----------------------------------------------------------------------------

// this is a matrix that broadcasts the same uniform f32m44 across all 8 lanes
struct f32m44x8 {
	f32v4x8 col[4];
};

// broadcasts the same matrix across all 8 lanes
inline f32m44x8 f32m44x8_broadcast(srat::array<f32, 16> const & m) {
	#define M(i) f32x8_splat(m[i])
	return {
		f32v4x8 { M(0), M(1), M(2), M(3) },
		f32v4x8 { M(4), M(5), M(6), M(7) },
		f32v4x8 { M(8), M(9), M(10), M(11) },
		f32v4x8 { M(12), M(13), M(14), M(15) },
	};
	#undef M
};

inline f32m44x8 f32m44x8_broadcast(f32m44 const & m) {
	return f32m44x8_broadcast(m.m);
};

// reads the matrix back into memory
inline void f32m44x8_store(
	f32m44x8 const & m, srat::array<f32, 16> & out
) {
	out[0]  = f32x8_lane0(m.col[0].v[0]);
	out[1]  = f32x8_lane0(m.col[0].v[1]);
	out[2]  = f32x8_lane0(m.col[0].v[2]);
	out[3]  = f32x8_lane0(m.col[0].v[3]);

	out[4]  = f32x8_lane0(m.col[1].v[0]);
	out[5]  = f32x8_lane0(m.col[1].v[1]);
	out[6]  = f32x8_lane0(m.col[1].v[2]);
	out[7]  = f32x8_lane0(m.col[1].v[3]);

	out[8]  = f32x8_lane0(m.col[2].v[0]);
	out[9]  = f32x8_lane0(m.col[2].v[1]);
	out[10] = f32x8_lane0(m.col[2].v[2]);
	out[11] = f32x8_lane0(m.col[2].v[3]);

	out[12] = f32x8_lane0(m.col[3].v[0]);
	out[13] = f32x8_lane0(m.col[3].v[1]);
	out[14] = f32x8_lane0(m.col[3].v[2]);
	out[15] = f32x8_lane0(m.col[3].v[3]);
}

inline f32v4x8 f32m44x8_mul_vec(f32m44x8 const & m, f32v4x8 const & v) {
	// column-major matrix-vector multiply
	f32v4x8 result = f32v4x8_zero();

	// result += m.col[0] * v.x
	result.x = f32x8_fmadd(m.col[0].v[0], v.x, result.x);
	result.y = f32x8_fmadd(m.col[0].v[1], v.x, result.y);
	result.z = f32x8_fmadd(m.col[0].v[2], v.x, result.z);
	result.w = f32x8_fmadd(m.col[0].v[3], v.x, result.w);

	// result += m.col[1] * v.y
	result.x = f32x8_fmadd(m.col[1].v[0], v.y, result.x);
	result.y = f32x8_fmadd(m.col[1].v[1], v.y, result.y);
	result.z = f32x8_fmadd(m.col[1].v[2], v.y, result.z);
	result.w = f32x8_fmadd(m.col[1].v[3], v.y, result.w);

	// result += m.col[2] * v.z
	result.x = f32x8_fmadd(m.col[2].v[0], v.z, result.x);
	result.y = f32x8_fmadd(m.col[2].v[1], v.z, result.y);
	result.z = f32x8_fmadd(m.col[2].v[2], v.z, result.z);
	result.w = f32x8_fmadd(m.col[2].v[3], v.z, result.w);

	// result += m.col[3] * v.w
	result.x = f32x8_fmadd(m.col[3].v[0], v.w, result.x);
	result.y = f32x8_fmadd(m.col[3].v[1], v.w, result.y);
	result.z = f32x8_fmadd(m.col[3].v[2], v.w, result.z);
	result.w = f32x8_fmadd(m.col[3].v[3], v.w, result.w);

	return result;
}

// -----------------------------------------------------------------------------
// -- f32 math
// -----------------------------------------------------------------------------

inline f32 f32_min(f32 const a, f32 const b) { return (a < b) ? a : b; }
inline f32 f32_max(f32 const a, f32 const b) { return (a > b) ? a : b; }
inline f32 f32_abs(f32 const a) { return (a < 0.0f) ? -a : a; }
inline f32 f32_round(f32 const a) { return (a >= 0.0f) ? f32(i32(a + 0.5f)) : f32(i32(a - 0.5f)); }
inline f32 f32_clamp(f32 const v, f32 const min, f32 const max) {
	return f32_min(f32_max(v, min), max);
}

// -----------------------------------------------------------------------------
// -- i32 bounding box
// -----------------------------------------------------------------------------

struct i32bbox2 {
	i32v2 min;
	i32v2 max;
};

inline i32bbox2 i32bbox2_from_triangle(
	i32v2 const & v0,
	i32v2 const & v1,
	i32v2 const & v2
) {
	return i32bbox2 {
		.min = {
			.x = i32_min(i32_min(v0.x, v1.x), v2.x),
			.y = i32_min(i32_min(v0.y, v1.y), v2.y),
		},
		.max = {
			.x = i32_max(i32_max(v0.x, v1.x), v2.x),
			.y = i32_max(i32_max(v0.y, v1.y), v2.y),
		}
	};
}

// -----------------------------------------------------------------------------
// -- f32 bounding box
// -----------------------------------------------------------------------------

struct f32bbox2 {
	f32v2 min;
	f32v2 max;
};

inline f32bbox2 f32bbox2_from_triangle(
	f32v2 const & v0,
	f32v2 const & v1,
	f32v2 const & v2
) {
	return f32bbox2 {
		.min = {
			.x = f32_min(f32_min(v0.x, v1.x), v2.x),
			.y = f32_min(f32_min(v0.y, v1.y), v2.y),
		},
		.max = {
			.x = f32_max(f32_max(v0.x, v1.x), v2.x),
			.y = f32_max(f32_max(v0.y, v1.y), v2.y),
		}
	};
}

// -----------------------------------------------------------------------------
// -- rasterizer specific math
// -----------------------------------------------------------------------------

[[nodiscard]] inline f32 topLeftRuleBias(
	i32v2 const & a,
	i32v2 const & b
) {
	// top edge is horizontal and goes left,
	// left edge is non-horizontal and runs downwards
	if ((a.y == b.y && a.x > b.x) || a.y < b.y) {
		return 0.0f;
	}
	return -0.5f;
}

// computes triangle edge function
// (bx-ax)*(py-ay) - (by-ay)*(px-ax)
inline f32x8 f32x8_barycentric(
	f32v2 const & a,
	f32v2 const & b,
	f32x8 const & px,
	f32x8 const & py
) {
	f32x8 const dbx = f32x8_splat(b.x - a.x);
	f32x8 const dby = f32x8_splat(b.y - a.y);
	f32x8 const dpx = px + f32x8_splat(-a.x);
	f32x8 const dpy = py + f32x8_splat(-a.y);
	return f32x8_fmadd(dbx, dpy, -(dby * dpx));
}

inline f32 f32v2_triangle_parallelogram_area(
	f32v2 const & v0,
	f32v2 const & v1,
	f32v2 const & v2
) {
	return (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
}

inline i32v2 f32v4_clip_to_screen(f32v4 const & v, u32v2 const & screenSize) {
	return i32v2 {
		.x = (i32)(( v.x/v.w * 0.5f + 0.5f) * (f32)screenSize.x),
		.y = (i32)(( v.y/v.w * 0.5f + 0.5f) * (f32)screenSize.y),
	};
};

inline f32 depth_ndc(f32v4 const & v) {
	return v.z / v.w * 0.5f + 0.5f;
};

// -----------------------------------------------------------------------------
// -- conversion
// -----------------------------------------------------------------------------

inline f32v2 as_f32v2(i32v2 const & v) {
	return f32v2 { (f32)v.x, (f32)v.y };
};

inline u32 as_rgba(f32v4 const & color) {
	u32 r = (u32)(f32_clamp(color.x, 0.0f, 1.0f) * 255.0f + 0.5f);
	u32 g = (u32)(f32_clamp(color.y, 0.0f, 1.0f) * 255.0f + 0.5f);
	u32 b = (u32)(f32_clamp(color.z, 0.0f, 1.0f) * 255.0f + 0.5f);
	u32 a = (u32)(f32_clamp(color.w, 0.0f, 1.0f) * 255.0f + 0.5f);
	return (a << 24) | (b << 16) | (g << 8) | r;
}

// -----------------------------------------------------------------------------
// -- free form funcs
// -----------------------------------------------------------------------------
// all above free form funcs should be moved to here

[[nodiscard]] inline i32x8 f32x8_to_i32x8_trunc(f32x8 const & v) {
	return { _mm256_cvttps_epi32(v.v) };
}

[[nodiscard]] inline i32x8 f32x8_to_i32x8_round(f32x8 const & v) {
	return { _mm256_cvtps_epi32(v.v) };
}
[[nodiscard]] inline i32v2x8 f32v2x8_to_i32v2x8_round(f32v2x8 const & v) {
	return {
		f32x8_to_i32x8_round(v.x),
		f32x8_to_i32x8_round(v.y),
	};
}

[[nodiscard]] inline i32x8 i32x8_clamp(
	i32x8 const & v, i32x8 const & min, i32x8 const & max
) {
	return {
		_mm256_max_epi32(min.v, _mm256_min_epi32(max.v, v.v))
	};
}
[[nodiscard]] inline i32v2x8 i32v2x8_clamp(
	i32v2x8 const & v, i32v2x8 const & min, i32v2x8 const & max
) {
	return {
		i32x8_clamp(v.x, min.x, max.x),
		i32x8_clamp(v.y, min.y, max.y),
	};
}
[[nodiscard]] inline i32x8 i32x8_floor(f32x8 const & v) {
	return {
		_mm256_cvttps_epi32(_mm256_floor_ps(v.v))
	};
}
[[nodiscard]] inline f32x8 f32x8_floor(f32x8 const & v)
	{ return { _mm256_floor_ps(v.v) }; }

[[nodiscard]] inline f32x8 f32x8_trunc(f32x8 const & v)
	{ return { _mm256_round_ps(v.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) }; }

[[nodiscard]] inline f32v2x8 f32v2x8_floor(f32v2x8 const & v)
	{ return { f32x8_floor(v.x), f32x8_floor(v.y), }; }

[[nodiscard]] inline f32v2x8 f32v2x8_trunc(f32v2x8 const & v)
	{ return { f32x8_trunc(v.x), f32x8_trunc(v.y), }; }


// a*b + c
[[nodiscard]] inline f32x8 f32x8_madd(
	f32x8 const & a, f32x8 const & b, f32x8 const & c
) {
	return { _mm256_fmadd_ps(a.v, b.v, c.v) };
}
[[nodiscard]] inline f32v2x8 f32v2x8_madd(
	f32v2x8 const & a, f32v2x8 const & b, f32v2x8 const & c
) {
	return { f32x8_madd(a.x, b.x, c.x), f32x8_madd(a.y, b.y, c.y), };
}

// a - b*c
[[nodiscard]] inline f32x8 f32x8_nmadd(
	f32x8 const & a, f32x8 const & b, f32x8 const & c
) {
	return { _mm256_fnmadd_ps(a.v, b.v, c.v) };
}
[[nodiscard]] inline f32v2x8 f32v2x8_nmadd(
	f32v2x8 const & a, f32v2x8 const & b, f32v2x8 const & c
) {
	return { f32x8_nmadd(a.x, b.x, c.x), f32x8_nmadd(a.y, b.y, c.y), };
}

[[nodiscard]] inline f32x8 f32x8_modulo(f32x8 const & a, f32x8 const & b) {
	f32x8 const div = a / b;
	f32x8 const divFloor = f32x8_floor(div);
	return f32x8 { f32x8_nmadd(divFloor, b, a) };
}

[[nodiscard]] inline f32v2x8 f32v2x8_modulo(
	f32v2x8 const & a, f32v2x8 const & b
) {
	return { f32x8_modulo(a.x, b.x), f32x8_modulo(a.y, b.y), };
}

[[nodiscard]] inline i32v2x8 i32v2x8_splat(i32 const x, i32 const y) {
	return {
		i32x8_splat(x),
		i32x8_splat(y),
	};
}

[[nodiscard]] inline f32x8 f32x8_memory_gather(
	float const * const address,
	i32x8 const & offsets
) {
	return { _mm256_i32gather_ps(address, offsets.v, 4) };
}

[[nodiscard]] inline i32v2x8 f32v2x8_to_i32v2x8_floor(f32v2x8 const & v) {
	return {
		f32x8_to_i32x8_trunc(f32x8_floor(v.x)),
		f32x8_to_i32x8_trunc(f32x8_floor(v.y)),
	};
}
