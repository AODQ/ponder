import "util-material-openpbr-load";
struct PC { u32v4 pad[8]; u64 outVa; };
[[vk::push_constant]] PC pc;
[shader("compute")]
[numthreads(64,1,1)]
void main(const u32v3 tid : SV_DispatchThreadID) { ((f32 *)pc.outVa)[tid.x] = 0.0f; }
