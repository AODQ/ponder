#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

layout(set = 0, binding = 0) uniform sampler2D vkofTextures[];

#include "shared/scene.h"

layout(location = 0) in flat uint inVisibility;
layout(location = 0) out uint outColor;

void main() {
	// mask is: modelId: 8 bits, meshlet: 17 bits, triangle: 7 bits
	// visbility has modelId and meshlet, just need to mask in triangleId
	outColor = inVisibility | (gl_PrimitiveID & 0x7Fu);
}
