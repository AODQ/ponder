#pragma once
__exported import "util-types";
__exported import "util-random";
__exported import "util-regir";

struct PC {
	u32v4 rootPad[8];
	uint64_t outVa;
	uint64_t inVa;
	uint64_t guardsVa;
	uint64_t cellsVa;
	uint count;
	// table capacity in slots, not live cells
	uint cellCount;
	uint mode;
	f32 cellSize;
};

[[vk::push_constant]] PC pc;

// shared prelude for the regir tier-9 shaders (plan-many-lights.md section 9).
// one push-constant layout across every regir dispatch so a single C++ push
// struct drives all of them, matching the util-emitter-test.glsl convention.



// a query point; position and normal are independent so the same buffer
// drives both the grid-coordinate and the normal-bucket cases
public struct RegirTestPoint {
	public f32v3 position;
	public f32v3 normal;
};


public static const u32 REGIR_TEST_MODE_KEY = 0u;
public static const u32 REGIR_TEST_MODE_INSERT = 1u;
public static const u32 REGIR_TEST_MODE_LOOKUP = 2u;
// mode key: gridCoord.xyz (bit-cast i32), normalBucket, hashKey
public static const u32 REGIR_TEST_KEY_STRIDE = 5u;
// modes insert/lookup: slot, probeCount
public static const u32 REGIR_TEST_SLOT_STRIDE = 2u;