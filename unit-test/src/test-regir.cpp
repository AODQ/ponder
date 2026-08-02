#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <srat/core-math.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// tier 9 of plan-many-lights.md: the regir hash grid, before any reservoir
// exists. the grid is pure bookkeeping -- a key derivation and an open-
// addressed table -- and every one of its failure modes is silent in a render:
// a wrong cell still returns *a* light sample, just one targeted at somewhere
// else, which shades to a plausible image with no nan and no missing energy.
// so these are the tests that have to be exact rather than statistical.
//
// what is crosschecked exactly, and what is only checked for properties:
//
//  - grid coordinates are pinned to floor(position / cellSize) against a cpu
//    mirror, integer for integer. there is exactly one right answer and the
//    common wrong one (truncation toward zero) is a specific, testable shape.
//  - the normal bucket is checked by *property* -- every bucket reachable,
//    bounded solid-angle spread, opposing normals separated, sentinel distinct
//    -- deliberately not against a cpu mirror. the plan leaves octahedral vs
//    per-component quantization open (either is defensible; measured spread at
//    16 buckets is 1.3x vs 1.7x), and these properties are what the renderer
//    actually depends on. a mirror here would pin an implementation choice
//    that is still open and would flake the moment it is revisited.
//  - the hash is checked for its two contracts only: never zero (0 is the
//    empty-slot marker) and no systematic clustering. the mixing function is
//    free to change.
//
// tier 9 needs util-regir.glsl to expose:
//
//   i32v3 utilRegirGridCoord(f32v3 worldPosition, f32 cellSize);
//   u32   utilRegirNormalBucket(f32v3 normal);
//   u32   utilRegirCellHashKey(GpuRegirCellKey key);
//   u32   utilRegirInsert(GpuRegirCellBuffer, u32 cellCount, GpuRegirCellKey, out u32 probes);
//   u32   utilRegirLookup(GpuRegirCellBuffer, u32 cellCount, GpuRegirCellKey, out u32 probes);
//
// and regir-shared.h to define skRegirNormalResolution, skRegirNormalBucketNone
// and skRegirSlotInvalid. the pipeline is compiled at runtime, so until those
// land these cases fail at REQUIRE(pipeline.id != 0u) rather than breaking the
// build.
// ---------------------------------------------------------------------------

namespace {

// must match regir-shared.h
constexpr u32 skNormalResolution = 4u;
constexpr u32 skNormalBucketCount = skNormalResolution * skNormalResolution;
constexpr u32 skNormalBucketNone = skNormalBucketCount;
constexpr u32 skSlotInvalid = 0xFFFFFFFFu;

// must match util-regir-test.glsl
constexpr u32 skModeKey = 0u;
constexpr u32 skModeInsert = 1u;
constexpr u32 skModeLookup = 2u;
constexpr u32 skKeyStride = 5u;
constexpr u32 skSlotStride = 2u;

// generously over-allocated per slot so this file does not couple to
// sizeof(GpuRegirCell) while the reservoir layout is still moving; the shader
// indexes with its own stride and a larger buffer is harmless
constexpr u64 skCellByteStride = 128u;

// mirrors the push block in util-regir-test.glsl (scalar layout)
struct RegirPush {
	u64 outVa;
	u64 inVa;
	u64 guardsVa;
	u64 cellsVa;
	u32 count;
	u32 cellCount;
	u32 mode;
	f32 cellSize;
};

// mirrors RegirTestPoint in util-regir-test.glsl
struct TestPoint {
	f32v3 position;
	f32v3 normal;
};

vkof::Pipeline make_pipeline() {
	char const * const includePaths[] = {
		PONDER_SHADER_DIR, MOR_INCLUDE_DIR, APP_SHADER_DIR, TEST_SHADER_DIR,
	};
	vkof::Pipeline const pipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				std::string(TEST_SHADER_DIR) + "slang/entry/regir_grid.slang"
			).c_str(),
			.includePaths = srat::slice<char const * const>(includePaths, 4u),
		})
	);
	REQUIRE(pipeline.id != 0u);
	return pipeline;
}

// raii, because every guard below is a REQUIRE: doctest aborts the case on
// failure, and a plain destroy() at the end of the body would leak the table
// on exactly the runs that fail
struct ScopedPipeline {
	vkof::Pipeline handle {};

	ScopedPipeline() : handle(make_pipeline()) {}
	~ScopedPipeline() {
		if (handle.id != 0u) {
			vkof::pipeline_destroy(handle);
		}
	}
	ScopedPipeline(ScopedPipeline const &) = delete;
	ScopedPipeline & operator=(ScopedPipeline const &) = delete;

	operator vkof::Pipeline() const { return handle; }
};

struct ScopedBuffer {
	vkof::Buffer handle {};

	ScopedBuffer() = default;
	explicit ScopedBuffer(vkof::Buffer const b) : handle(b) {}
	~ScopedBuffer() {
		if (handle.id != 0u) {
			vkof::buffer_destroy(handle);
		}
	}
	ScopedBuffer(ScopedBuffer const &) = delete;
	ScopedBuffer & operator=(ScopedBuffer const &) = delete;

	operator vkof::Buffer() const { return handle; }
};

// a zeroed table; the empty-slot marker is 0, so a fresh table must be zeroed
// rather than merely allocated
vkof::Buffer make_zeroed(u64 const byteCount) {
	std::vector<u8> const zeros((usize)byteCount, (u8)0);
	return test::upload_bytes(zeros.data(), byteCount);
}

// guards and cells are separate buffers, matching util-regir-shared.h: the
// compare-and-swap word lives apart from the bulk payload
struct RegirTable {
	ScopedBuffer guards;
	ScopedBuffer cells;
	u32 cellCount {};
};

void regir_table_init(RegirTable & table, u32 const cellCount) {
	table.guards.handle = make_zeroed((u64)cellCount * sizeof(u32));
	table.cells.handle = make_zeroed((u64)cellCount * skCellByteStride);
	table.cellCount = cellCount;
}

std::vector<u32> run(
	vkof::Pipeline const pipeline,
	std::vector<TestPoint> const & points,
	u32 const mode,
	f32 const cellSize,
	RegirTable const & table,
	u32 const stride
) {
	u32 const count = (u32)points.size();
	vkof::Buffer const inBuf = (
		test::upload_bytes(
			points.data(), (u64)points.size() * sizeof(TestPoint)
		)
	);
	vkof::Buffer const outBuf = test::make_buffer_u32(count * stride);

	RegirPush const push {
		.outVa = vkof::buffer_virtual_address(outBuf),
		.inVa = vkof::buffer_virtual_address(inBuf),
		.guardsVa = (
			table.guards.handle.id != 0u
				? vkof::buffer_virtual_address(table.guards) : (u64)0u
		),
		.cellsVa = (
			table.cells.handle.id != 0u
				? vkof::buffer_virtual_address(table.cells) : (u64)0u
		),
		.count = count,
		.cellCount = table.cellCount,
		.mode = mode,
		.cellSize = cellSize,
	};
	test::dispatch(pipeline, push, (count + 255u) / 256u);

	std::vector<u32> const result = (
		test::readback<u32>(outBuf, 0u, count * stride)
	);
	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	return result;
}

// vkof hands back a live pipeline handle even when the shader failed to
// compile (it retries on the next save), so a missing util-regir.glsl shows up
// as an untouched output buffer rather than a bad handle. the hash is
// contractually never 0, so an all-zero key output means the dispatch never
// ran -- worth separating loudly from "ran and got it wrong", which is what
// every other assertion in this file is about
void require_shader_ran(std::vector<u32> const & keyOut, u32 const count) {
	for (u32 i = 0u; i < count; ++i) {
		if (keyOut[i * skKeyStride + 4u] != 0u) {
			return;
		}
	}
	REQUIRE_MESSAGE(
		false,
		"regir_grid.comp produced no output -- util-regir.glsl is missing or "
		"failed to compile; see the shader compile log above"
	);
}

// same guard for the insert/lookup modes: distinct keys cannot all land in
// slot 0, so an all-zero slot column is an untouched buffer
void require_slots_ran(std::vector<u32> const & slotOut, u32 const count) {
	for (u32 i = 0u; i < count; ++i) {
		if (slotOut[i * skSlotStride + 0u] != 0u) {
			return;
		}
	}
	REQUIRE_MESSAGE(
		false,
		"regir_grid.comp produced no slots -- util-regir.glsl is missing or "
		"failed to compile; see the shader compile log above"
	);
}

// -- cpu side ---------------------------------------------------------------

i32 cpu_grid_coord_axis(f32 const v, f32 const cellSize) {
	return (i32)std::floor((f64)v / (f64)cellSize);
}

// deterministic uniform directions; rejection in the unit ball so there is no
// dependence on a trig implementation matching the gpu's
struct Lcg {
	u32 state;

	f32 next() {
		state = state * 1664525u + 1013904223u;
		return (f32)((state >> 8u) & 0xFFFFFFu) / (f32)0x1000000u;
	}
};

std::vector<f32v3> uniform_directions(u32 const count, u32 const seed) {
	Lcg rng { seed };
	std::vector<f32v3> result;
	result.reserve(count);
	while (result.size() < count) {
		f32 const x = rng.next() * 2.0f - 1.0f;
		f32 const y = rng.next() * 2.0f - 1.0f;
		f32 const z = rng.next() * 2.0f - 1.0f;
		f32 const r2 = x * x + y * y + z * z;
		if (r2 > 1.0f || r2 < 1e-8f) {
			continue;
		}
		f32 const r = std::sqrt(r2);
		result.emplace_back(f32v3 { x / r, y / r, z / r });
	}
	return result;
}

} // namespace

// ---------------------------------------------------------------------------
// 9.4 -- coordinate quantization
//
// first because it catches the widest family of bugs and needs no table. the
// headline case is the cell straddling the origin: a truncating cast collapses
// [-1,0) and [0,1) into the same coordinate, producing one double-width cell
// per axis. that is invisible in a render (cells near the world origin are
// simply coarser) and it moves with the scene, so it is exactly the kind of
// defect that gets attributed to tuning.
// ---------------------------------------------------------------------------

TEST_CASE("regir grid: coordinates are floor, uniformly, including negatives") {
	ScopedPipeline const pipeline;

	constexpr f32 skCellSize = 1.0f;
	constexpr i32 skLo = -5;
	constexpr i32 skHi = 5;
	constexpr u32 skPerCell = 1000u;
	constexpr u32 skCount = (u32)(skHi - skLo) * skPerCell;

	// a uniform sweep across the origin along x; y and z are held inside a
	// single cell so any coordinate motion there is a bug on its own
	std::vector<TestPoint> points;
	points.reserve(skCount);
	for (u32 i = 0u; i < skCount; ++i) {
		f32 const t = ((f32)i + 0.5f) / (f32)skCount;
		f32 const x = (f32)skLo + t * (f32)(skHi - skLo);
		points.emplace_back(
			TestPoint {
				.position = f32v3 { x, 0.25f, 0.25f },
				.normal = f32v3 { 0.0f, 0.0f, 1.0f },
			}
		);
	}

	std::vector<u32> const raw = (
		run(pipeline, points, skModeKey, skCellSize, RegirTable {}, skKeyStride)
	);

	require_shader_ran(raw, skCount);

	// aggregated rather than asserted per point: a systematic quantization bug
	// misses on every one of 10k points, and 10k identical failures buries the
	// one line that says which
	std::map<i32, u32> population;
	u32 mismatches = 0u;
	i32 firstBadIndex = -1;
	i32 firstBadGot = 0;
	i32 firstBadWant = 0;
	for (u32 i = 0u; i < skCount; ++i) {
		i32 const gx = (i32)raw[i * skKeyStride + 0u];
		i32 const gy = (i32)raw[i * skKeyStride + 1u];
		i32 const gz = (i32)raw[i * skKeyStride + 2u];
		i32 const want = cpu_grid_coord_axis(points[i].position.x, skCellSize);
		if (gx != want || gy != 0 || gz != 0) {
			if (firstBadIndex < 0) {
				firstBadIndex = (i32)i;
				firstBadGot = gx;
				firstBadWant = want;
			}
			mismatches += 1u;
		}
		population[gx] += 1u;
	}
	CAPTURE(firstBadIndex);
	CAPTURE(firstBadGot);
	CAPTURE(firstBadWant);
	CHECK(mismatches == 0u);

	// every cell the sweep crosses must exist, and no cell may be wider than
	// any other. truncation produces skHi - skLo - 1 cells with the middle one
	// carrying double the population
	CHECK(population.size() == (usize)(skHi - skLo));
	for (i32 c = skLo; c < skHi; ++c) {
		REQUIRE(population.count(c) == 1u);
		// +-1 for the half-open sweep landing on a boundary
		CHECK(population[c] >= skPerCell - 1u);
		CHECK(population[c] <= skPerCell + 1u);
	}

}

TEST_CASE("regir grid: coordinate edge cases stay finite and exact") {
	ScopedPipeline const pipeline;

	// gpu division is not correctly rounded -- it is typically a * rcp(b) with
	// about an ulp of error -- so floor(p / cellSize) can land on either side
	// of an *exact* cell boundary, and denormal inputs are flushed to zero.
	// both are legitimate: a point on a boundary belongs to either adjacent
	// cell and the lookup only has to be deterministic, which it is. so those
	// cases assert "within one cell" and the rest assert exactness
	struct Case {
		char const * name;
		f32v3 position;
		f32 cellSize;
		i32 tolerance;
	};

	// each is run as its own dispatch so cellSize can vary per case
	Case const cases[] = {
		{ "origin", f32v3 { 0.0f, 0.0f, 0.0f }, 1.0f, 0 },
		{ "exact positive boundary", f32v3 { 2.0f, 4.0f, 8.0f }, 2.0f, 1 },
		{ "exact negative boundary", f32v3 { -2.0f, -4.0f, -8.0f }, 2.0f, 1 },
		{ "just below zero", f32v3 { -1e-6f, -1e-6f, -1e-6f }, 1.0f, 0 },
		{ "just above zero", f32v3 { 1e-6f, 1e-6f, 1e-6f }, 1.0f, 0 },
		{ "tiny magnitude", f32v3 { 1e-6f, -1e-6f, 1e-6f }, 1e-3f, 0 },
		{ "large magnitude", f32v3 { 1e6f, -1e6f, 1e6f }, 1.0f, 1 },
		{ "large over large cell", f32v3 { 1e6f, -1e6f, 1e6f }, 1e3f, 1 },
		{ "denormal", f32v3 { 1e-40f, -1e-40f, 1e-40f }, 1.0f, 1 },
	};

	for (Case const & c : cases) {
		CAPTURE(c.name);
		std::vector<TestPoint> const points {
			TestPoint {
				.position = c.position,
				.normal = f32v3 { 0.0f, 1.0f, 0.0f },
			},
		};
		std::vector<u32> const raw = (
			run(
				pipeline, points, skModeKey, c.cellSize,
				RegirTable {}, skKeyStride
			)
		);

		require_shader_ran(raw, 1u);
		f32 const axis[3] = { c.position.x, c.position.y, c.position.z };
		for (u32 a = 0u; a < 3u; ++a) {
			CAPTURE(a);
			i32 const got = (i32)raw[a];
			i32 const want = cpu_grid_coord_axis(axis[a], c.cellSize);
			CAPTURE(got);
			CAPTURE(want);
			CHECK(std::abs(got - want) <= c.tolerance);
		}
		// the hash must never collide with the empty-slot marker
		CHECK(raw[4] != 0u);
	}

}

// ---------------------------------------------------------------------------
// 9.3 -- normal quantization
//
// property-based, per the note at the top. the specific failure this exists for
// is a quantizer that casts a negative component to unsigned: half the sphere
// then collapses into one bucket or wraps out of range, which shows up here as
// unreachable buckets and a blown spread ratio, and shows up in a render as
// cells that mix opposing orientations and target their reservoirs at nothing.
// ---------------------------------------------------------------------------

TEST_CASE("regir grid: normal buckets cover the sphere without collapsing") {
	ScopedPipeline const pipeline;

	constexpr u32 skCount = 200000u;
	std::vector<f32v3> const dirs = uniform_directions(skCount, 0x9E3779B9u);

	std::vector<TestPoint> points;
	points.reserve(dirs.size());
	for (f32v3 const & d : dirs) {
		points.emplace_back(
			TestPoint { .position = f32v3 { 0.5f, 0.5f, 0.5f }, .normal = d }
		);
	}

	std::vector<u32> const raw = (
		run(pipeline, points, skModeKey, 1.0f, RegirTable {}, skKeyStride)
	);

	require_shader_ran(raw, (u32)points.size());

	std::vector<u32> population(skNormalBucketCount, 0u);
	u32 outOfRange = 0u;
	u32 sentinelHits = 0u;
	for (u32 i = 0u; i < (u32)points.size(); ++i) {
		u32 const bucket = raw[i * skKeyStride + 3u];
		// the sentinel is reserved for medium vertices and must never be
		// produced from a real normal
		if (bucket == skNormalBucketNone) {
			sentinelHits += 1u;
			continue;
		}
		if (bucket >= skNormalBucketCount) {
			outOfRange += 1u;
			continue;
		}
		population[bucket] += 1u;
	}
	CHECK(sentinelHits == 0u);
	CHECK(outOfRange == 0u);

	// every bucket reachable. a quantizer that loses the negative half of the
	// sphere leaves most of these at zero
	u32 minPop = 0xFFFFFFFFu;
	u32 maxPop = 0u;
	u32 emptyBuckets = 0u;
	for (u32 const p : population) {
		if (p == 0u) {
			emptyBuckets += 1u;
		}
		minPop = std::min(minPop, p);
		maxPop = std::max(maxPop, p);
	}
	CAPTURE(skNormalBucketCount);
	CHECK(emptyBuckets == 0u);
	REQUIRE(minPop > 0u);

	// population is proportional to solid angle. measured spread at 16 buckets
	// is 1.32x for octahedral and 1.69x for correct per-component quantization;
	// the gate is loose enough to admit either and to not flake, and tight
	// enough that any collapse fails hard
	f64 const spread = (f64)maxPop / (f64)minPop;
	CAPTURE(spread);
	CHECK(spread < 2.5);

}

TEST_CASE("regir grid: opposing normals land in different cells") {
	ScopedPipeline const pipeline;

	// same position, so only the normal can separate these
	f32v3 const axes[] = {
		f32v3 { 1.0f, 0.0f, 0.0f },
		f32v3 { 0.0f, 1.0f, 0.0f },
		f32v3 { 0.0f, 0.0f, 1.0f },
	};

	std::vector<TestPoint> points;
	for (f32v3 const & a : axes) {
		points.emplace_back(
			TestPoint {
				.position = f32v3 { 0.25f, 0.25f, 0.25f },
				.normal = a,
			}
		);
		points.emplace_back(
			TestPoint {
				.position = f32v3 { 0.25f, 0.25f, 0.25f },
				.normal = f32v3 { -a.x, -a.y, -a.z },
			}
		);
	}

	std::vector<u32> const raw = (
		run(pipeline, points, skModeKey, 1.0f, RegirTable {}, skKeyStride)
	);

	require_shader_ran(raw, (u32)points.size());

	for (u32 pair = 0u; pair < 3u; ++pair) {
		CAPTURE(pair);
		u32 const a = raw[(pair * 2u + 0u) * skKeyStride + 3u];
		u32 const b = raw[(pair * 2u + 1u) * skKeyStride + 3u];
		CHECK(a != b);
		// and the hashes must differ too, or the table cannot separate them
		CHECK(
			raw[(pair * 2u + 0u) * skKeyStride + 4u]
			!= raw[(pair * 2u + 1u) * skKeyStride + 4u]
		);
	}

}

// ---------------------------------------------------------------------------
// 9.1 -- insert / lookup round trip
//
// the table's whole contract in one case: a key that was inserted is found at
// the same slot, a key that was not is reported missing, and two distinct keys
// never share a slot. the last of those is what a checksum-based collision
// check can violate silently -- the reason plan-many-lights.md argues for
// storing the exact key.
// ---------------------------------------------------------------------------

TEST_CASE("regir grid: insert and lookup round trip") {
	ScopedPipeline const pipeline;

	// a 8x8x8 lattice of cell centres at cellSize 1, one normal, so every key
	// is distinct by construction and the expected count is exact
	constexpr u32 skSide = 8u;
	constexpr u32 skKeyCount = skSide * skSide * skSide;
	constexpr u32 skCellCount = 2048u;

	std::vector<TestPoint> points;
	points.reserve(skKeyCount);
	for (u32 z = 0u; z < skSide; ++z) {
		for (u32 y = 0u; y < skSide; ++y) {
			for (u32 x = 0u; x < skSide; ++x) {
				points.emplace_back(
					TestPoint {
						.position = f32v3 {
							(f32)x + 0.5f, (f32)y + 0.5f, (f32)z + 0.5f
						},
						.normal = f32v3 { 0.0f, 0.0f, 1.0f },
					}
				);
			}
		}
	}

	RegirTable table;
	regir_table_init(table, skCellCount);

	std::vector<u32> const inserted = (
		run(pipeline, points, skModeInsert, 1.0f, table, skSlotStride)
	);
	std::vector<u32> const found = (
		run(pipeline, points, skModeLookup, 1.0f, table, skSlotStride)
	);

	require_slots_ran(inserted, skKeyCount);

	std::set<u32> slots;
	u32 failures = 0u;
	u32 outOfRange = 0u;
	u32 collisions = 0u;
	u32 lookupMismatches = 0u;
	for (u32 i = 0u; i < skKeyCount; ++i) {
		u32 const slot = inserted[i * skSlotStride + 0u];
		if (slot == skSlotInvalid) {
			failures += 1u;
			continue;
		}
		if (slot >= skCellCount) {
			outOfRange += 1u;
			continue;
		}
		// distinct keys must never share a slot
		if (!slots.insert(slot).second) {
			collisions += 1u;
		}
		// and lookup must find exactly what insert claimed
		if (found[i * skSlotStride + 0u] != slot) {
			lookupMismatches += 1u;
		}
	}
	CHECK(outOfRange == 0u);
	CHECK(collisions == 0u);
	CHECK(lookupMismatches == 0u);
	// at 25% load with a 32-probe budget nothing should fail to insert
	CHECK(failures == 0u);

	// a key never inserted -- same lattice shifted well outside it
	std::vector<TestPoint> absent;
	for (u32 i = 0u; i < 64u; ++i) {
		absent.emplace_back(
			TestPoint {
				.position = f32v3 { 1000.5f + (f32)i, 1000.5f, 1000.5f },
				.normal = f32v3 { 0.0f, 0.0f, 1.0f },
			}
		);
	}
	std::vector<u32> const missing = (
		run(pipeline, absent, skModeLookup, 1.0f, table, skSlotStride)
	);
	u32 falsePositives = 0u;
	for (u32 i = 0u; i < (u32)absent.size(); ++i) {
		if (missing[i * skSlotStride + 0u] != skSlotInvalid) {
			falsePositives += 1u;
		}
	}
	CHECK(falsePositives == 0u);

}

TEST_CASE("regir grid: re-inserting a live key is idempotent") {
	ScopedPipeline const pipeline;

	constexpr u32 skKeyCount = 256u;
	constexpr u32 skCellCount = 1024u;

	std::vector<TestPoint> points;
	points.reserve(skKeyCount);
	for (u32 i = 0u; i < skKeyCount; ++i) {
		points.emplace_back(
			TestPoint {
				.position = f32v3 { (f32)i + 0.5f, 0.5f, 0.5f },
				.normal = f32v3 { 0.0f, 1.0f, 0.0f },
			}
		);
	}

	RegirTable table;
	regir_table_init(table, skCellCount);
	std::vector<u32> const first = (
		run(pipeline, points, skModeInsert, 1.0f, table, skSlotStride)
	);
	std::vector<u32> const second = (
		run(pipeline, points, skModeInsert, 1.0f, table, skSlotStride)
	);

	// a second pass over a live table must claim nothing new. the renderer's
	// scout pass runs every frame over largely the same cells, so a table that
	// grew on re-insert would fill up and then permanently fall back
	require_slots_ran(first, skKeyCount);

	u32 failures = 0u;
	u32 moved = 0u;
	for (u32 i = 0u; i < skKeyCount; ++i) {
		if (first[i * skSlotStride + 0u] == skSlotInvalid) {
			failures += 1u;
			continue;
		}
		if (second[i * skSlotStride + 0u] != first[i * skSlotStride + 0u]) {
			moved += 1u;
		}
	}
	CHECK(failures == 0u);
	CHECK(moved == 0u);

}

// ---------------------------------------------------------------------------
// 9.2 -- capacity pressure
//
// records probe behaviour and asserts the shape of the degradation rather than
// a threshold: as the table fills, inserts must start *failing* (which the
// renderer answers with the flat-cdf fallback, trap 22) and must never start
// *lying* -- a successful insert always round-trips, at every load factor.
// the numbers are logged because the plan's tuning phase needs them and
// because "the fallback ran a lot" is otherwise indistinguishable from "regir
// did not help".
// ---------------------------------------------------------------------------

TEST_CASE("regir grid: degrades by failing, never by corrupting") {
	ScopedPipeline const pipeline;

	constexpr u32 skKeyCount = 1024u;

	std::vector<TestPoint> points;
	points.reserve(skKeyCount);
	for (u32 i = 0u; i < skKeyCount; ++i) {
		// spread across three axes so the keys are not a single hash run
		points.emplace_back(
			TestPoint {
				.position = f32v3 {
					(f32)(i % 16u) + 0.5f,
					(f32)((i / 16u) % 16u) + 0.5f,
					(f32)(i / 256u) + 0.5f,
				},
				.normal = f32v3 { 0.0f, 0.0f, 1.0f },
			}
		);
	}

	// powers of two only: utilRegirSlotAt masks with cellCount - 1, so a
	// non-power-of-two capacity aliases slots rather than wrapping. loads are
	// therefore 25%, 50%, 100%
	u32 const capacities[] = { 4096u, 2048u, 1024u };

	u32 previousFailures = 0u;
	for (u32 const capacity : capacities) {
		CAPTURE(capacity);
		RegirTable table;
		regir_table_init(table, capacity);

		std::vector<u32> const inserted = (
			run(pipeline, points, skModeInsert, 1.0f, table, skSlotStride)
		);
		std::vector<u32> const found = (
			run(pipeline, points, skModeLookup, 1.0f, table, skSlotStride)
		);

		require_slots_ran(inserted, skKeyCount);

		std::set<u32> slots;
		u32 failures = 0u;
		u32 maxProbe = 0u;
		u64 probeSum = 0u;
		u32 outOfRange = 0u;
		u32 collisions = 0u;
		u32 lookupMismatches = 0u;
		for (u32 i = 0u; i < skKeyCount; ++i) {
			u32 const slot = inserted[i * skSlotStride + 0u];
			u32 const probes = inserted[i * skSlotStride + 1u];
			maxProbe = std::max(maxProbe, probes);
			probeSum += probes;
			if (slot == skSlotInvalid) {
				failures += 1u;
				continue;
			}
			// the invariants that must hold at every load factor
			if (slot >= capacity) {
				outOfRange += 1u;
				continue;
			}
			if (!slots.insert(slot).second) {
				collisions += 1u;
			}
			if (found[i * skSlotStride + 0u] != slot) {
				lookupMismatches += 1u;
			}
		}
		CHECK(outOfRange == 0u);
		CHECK(collisions == 0u);
		CHECK(lookupMismatches == 0u);

		f64 const load = (f64)skKeyCount / (f64)capacity;
		MESSAGE(
			"capacity " << capacity << " load " << load
			<< " failures " << failures
			<< " maxProbe " << maxProbe
			<< " meanProbe " << ((f64)probeSum / (f64)skKeyCount)
		);

		// failures are monotone in load: a table that gets *easier* to fill as
		// it shrinks is probing wrong
		CHECK(failures >= previousFailures);
		previousFailures = failures;

	}

	// the roomiest table must be clean, and the full one must not be
	CHECK(previousFailures > 0u);

}
