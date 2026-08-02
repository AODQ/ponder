#pragma once

// nvidia ngx: dlss ray reconstruction. only built with PONDER_DLSS

#include <vkof/vkof.hpp>

#if defined(PONDER_DLSS)

namespace vkof
{
	// set at init: whether ray reconstruction can be created
	[[nodiscard]] bool ngx_rr_available();
	// why ray reconstruction is unavailable; empty when it is available
	[[nodiscard]] char const * ngx_rr_unavailable_reason();

	// mirrors NVSDK_NGX_RayReconstruction_Hint_Render_Preset
	enum struct NgxRrPreset : u32 {
		Default = 0u,
		D = 4u,
		E = 5u,
		F = 6u,
	};

	// mirrors NVSDK_NGX_PerfQuality_Value
	enum struct NgxRrQuality : u32 {
		Performance = 0u,
		Balanced = 1u,
		Quality = 2u,
		UltraPerformance = 3u,
		Dlaa = 5u,
	};

	// the render resolution ray reconstruction wants for an output resolution
	[[nodiscard]] u32v2 ngx_rr_render_size(
		NgxRrQuality const quality,
		u32v2 const outputSize
	);

	struct NgxRrEvaluateInfo {
		CommandBuffer cmd;
		// the inputs are read from their top-left renderSize texels
		u32v2 renderSize;
		u32v2 outputSize;
		NgxRrQuality quality;
		NgxRrPreset preset;
		// linear radiance
		Image color;
		Image diffuseAlbedo;
		Image specularAlbedo;
		// world-space shading normals
		Image normals;
		// linear roughness in r
		Image roughness;
		// linear view depth in r
		Image depth;
		// pixels from the current to the previous frame position
		Image motionVectors;
		// world distance from the primary hit along the specular ray
		Image specularHitDistance;
		// row-major, row vectors (v * m), unjittered; view space looks down +z
		f32 worldToView[16];
		f32 viewToClip[16];
		Image output;
		// render pixels, in [-0.5, 0.5]
		f32v2 jitter;
		bool reset;
	};

	// records an evaluation into cmd; the feature is (re)created on the first
	// call and when the sizes, quality or preset change. false when evaluation
	// failed
	bool ngx_rr_evaluate(NgxRrEvaluateInfo const & info);

	// releases the feature; waits for the device to go idle
	void ngx_rr_release();
}

#endif
