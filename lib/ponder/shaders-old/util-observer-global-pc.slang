// autogen from global-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-observer-global-pc";
__exported import "util-types";

#ifndef f32
#define i32v2 i32v2
#define i32v3 i32v3
#define i32v4 i32v4
#define f32v2 f32v2
#define f32v3 f32v3
#define f32v4 f32v4
#define f32m44 f32m4x4
#define u32v2 u32v2
#define u32v3 u32v3
#define u32v4 u32v4
#define u64 uint64_t
#endif // f32

// per-frame values that are too large for the 128-byte root pushconstant
public struct GpuGlobalExtended {
	public u32 renderWidth;
	public u32 renderHeight;
};

// root pushconstant, must be exactly 128 bytes so that each node has their
// own 128-byte pushconstant range at offset 128
public struct GpuGlobalPc {
	public f32 time;
	public f32v3 cameraPos;
	public f32 exposure;
	public f32 pad0;
	public f32m44 viewProj;
	// u64 extended; // VA(GpuGlobalExtended)
	public u64 models; // VA(GpuResolveModelIndirectBuffer)
	public u32 renderWidth;
	public u32 renderHeight;
	public u64 pad1;
	public u64 pad2;
	public u64 pad3;
};
