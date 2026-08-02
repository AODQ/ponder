// reads a struct uploaded by the cpu through a raw device address and echoes
// every field back. a layout disagreement between slang and the c++ mirror
// scrambles the values rather than failing to compile

struct LayoutProbe {
	uint64_t va;
	float3 color;
	uint flags;
	float value;
	uint pad;
};

struct PushConstants {
	uint4 rootPad[8];
	uint64_t probeVa;
	uint64_t outVa;
};

[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
	LayoutProbe * probe = (LayoutProbe *)pc.probeVa;
	uint * out = (uint *)pc.outVa;

	const LayoutProbe p = probe[0];
	out[0] = uint(p.va & 0xFFFFFFFFull);
	out[1] = uint(p.va >> 32ull);
	out[2] = asuint(p.color.x);
	out[3] = asuint(p.color.y);
	out[4] = asuint(p.color.z);
	out[5] = p.flags;
	out[6] = asuint(p.value);
	// second element, so the test also pins the array stride
	const LayoutProbe q = probe[1];
	out[7] = q.flags;
	out[8] = asuint(q.value);
}
