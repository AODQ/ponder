implementing types;

// mirrors GpuMaterialMedium in lib/ponder/src/material-media.cpp

namespace vkof {

public struct MaterialMediumRecord {
	// MediumType, None for a regular surface
	public u32 type;
	// PhaseFunctionType
	public u32 phase;
	public u32 priority;
	u32 pad0;
	public f32v3 σa;
	public f32 anisotropy;
	public f32v3 σs;
	public f32 draineAlpha;
	public f32 dropletDiameter;
	public f32 heightOrigin;
	public f32 heightFalloff;
	u32 pad1;
};

} // namespace vkof
