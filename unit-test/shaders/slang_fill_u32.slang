// slang counterpart of fill_u32.comp: writes through a raw device address,
// the slang spelling of glsl's buffer_reference

struct PushConstants {
	// root constants occupy [0,128)
	uint4 rootPad[8];
	uint64_t dstVa;
	uint value;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	if (tid.x >= pc.count) { return; }
	uint * dst = (uint *)pc.dstVa;
	dst[tid.x] = pc.value;
}
