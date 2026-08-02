// autogen from manifold-shade-pc.h by
// lib/tools/shared-header-to-slang.py

module "util-manifold-shade-pc";
__exported import "util-types";

public struct GpuManifoldShadePC {
	// BendPoint[width * height], written by manifold-probe.slang
	public u64 bendPointsVa;
	public u32 colorHandle;
	public u32 width;
	public u32 height;
	public u32 pad0;
	// added to every pixel that has a bend point
	public f32v3 debugColor;
	public u32 pad1;
};
