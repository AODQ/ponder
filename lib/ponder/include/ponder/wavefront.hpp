#pragma once

#include <ponder/regir.hpp>
#include <ponder/zeltner-tables.hpp>

#include <vkof/vkof.hpp>

#include <optional>

namespace ponder {

// one entry kernel's variants: spectrum and debug-path recording
struct WavefrontKernel {
	vkof::Pipeline spectral {};
	vkof::Pipeline rgb {};
	// compiled on first use by wavefront_debug_path_prepare
	vkof::Pipeline spectralDebug {};
	vkof::Pipeline rgbDebug {};
};

// paths walk a region in square tiles, one workgroup per tile
constexpr u32 skWavefrontTileSize = 16u;
constexpr u32 skWavefrontTileSlotCount = (
	skWavefrontTileSize * skWavefrontTileSize
);

// pixels, clamped to the image
struct WavefrontRegion {
	u32v2 origin;
	u32v2 extent;
};

struct Wavefront {
	WavefrontKernel camera {};
	WavefrontKernel trace {};
	// trace for paths within reach of the black hole
	WavefrontKernel traceCurved {};
	// shades the traced hit
	WavefrontKernel integrate {};
	// connects the shaded vertex's nee sample
	WavefrontKernel nee {};
	// writes each path's radiance to the output image
	WavefrontKernel output {};
	vkof::Pipeline accumulatePipeline {};

	// -- the pass in flight: one sample of every pixel
	// next step to dispatch, 0 when no pass is in flight
	u32 passStep {};
	u32 passSampleIndex {};
	u32 passRegirFillIndex {};
	bool passSpectral {};

	// -- path slots: a pass traces at most pathCapacity paths, from the slot
	// cursor to at most the region's end; one sweep of the region is a sample
	u32 pathCapacity {};
	u32 slotCursor {};
	WavefrontRegion passRegion {};
	u32 passSlotCount {};
	u32 passSlotFirst {};
	u32 passPathCount {};
	// 1 when the pass in flight ends a region sweep
	u32 passSweepCount {};
	// a debug pass traces from its pixel and leaves the cursor
	bool passHoldsCursor {};
	// slots still to overwrite since the last accumulation reset
	u32 resetRemaining {};

	// jakob-hanika 2019 rgb -> spectrum sigmoid coefficients (srgb gamut),
	// scale[resolution] then coefficients[3][resolution^3][3]
	vkof::Buffer rgb2specBuffer {};
	u64 rgb2specVa {};
	u32 rgb2specResolution {};

	vkof::Buffer globalBuffer {};
	u64 globalVa {};

	vkof::Buffer debugPathBuffer {};
	u64 debugPathVa {};

	// material lookup tables, bound through the global
	ZeltnerTables zeltnerTables {};

	vkof::Image accumXyzImage {};
	u32 accumXyzHandle {};

	// -- per-pixel path records, render-resolution
	vkof::Buffer transportStateBuffer {};
	u64 transportStateVa {};
	vkof::Buffer pathStateBuffer {};
	u64 pathStateVa {};
	vkof::Buffer hitRecordBuffer {};
	u64 hitRecordVa {};
	vkof::Buffer neeRecordBuffer {};
	u64 neeRecordVa {};
};

[[nodiscard]] Wavefront wavefront_create();
void wavefront_destroy(Wavefront & wavefront);

// (re)creates the render-resolution images and the path buffers, which hold
// pathBudget paths; call at startup, on resize and on budget change
void wavefront_resize(
	Wavefront & wavefront,
	u32 const width,
	u32 const height,
	u32 const pathBudget
);

struct WavefrontCamera {
	f32v3 position;
	f32v3 forward;
	f32 fovY;
};

// the camera's ray basis: right scaled by tan(fovY / 2) * aspect, up by
// tan(fovY / 2)
struct WavefrontCameraBasis {
	f32v3 forward;
	f32v3 right;
	f32v3 up;
};

[[nodiscard]] WavefrontCameraBasis wavefront_camera_basis(
	WavefrontCamera const & camera,
	f32 const aspect
);

struct WavefrontBlackhole {
	f32v3 ori;
	f32 mass;
	f32 spin;
	f32 lensRadius;
	u32 maxChords;
	f32 diskInnerRadius;
	f32 diskOuterRadius;
	f32 diskTemperature;
	f32 diskBrightness;
	f32 diskOpticalDepth;
	f32 diskOpticalDepthExponent;
	f32 stepFactor;
	f32 diskTime;
	f32 diskRotation;
	f32 diskNoiseAmplitude;
	f32 diskNoiseScale;
	f32 diskHotspotRadius;
	f32 diskHotspotTemperature;
	f32 diskHotspotSize;
};

// mirrors vkof::MediumType
enum struct WavefrontAmbientMediumType : u32 {
	None,
	Homogeneous,
	HeightFog,
};

// mirrors vkof::PhaseFunctionType
enum struct WavefrontAmbientMediumPhase : u32 {
	HenyeyGreenstein,
	Draine,
	Fog,
};

struct WavefrontAmbientMedium {
	WavefrontAmbientMediumType type;
	f32v3 sigmaAbsorption;
	f32v3 sigmaScattering;
	WavefrontAmbientMediumPhase phase;
	f32 anisotropy;
	f32 draineAlpha;
	f32 dropletDiameter;
	f32 heightOrigin;
	f32 heightFalloff;
};

// mirrors vkof::EnvironmentMode
enum struct WavefrontEnvironmentMode : u32 {
	Furnace,
	Checkerboard,
	Black,
	HdrMap,
};

struct WavefrontEnvironment {
	WavefrontEnvironmentMode mode;
	f32 intensity;
	f32 rotation;
	u32 radianceHandle;
	u32 pdfHandle;
	u32 aliasProbHandle;
	u32 aliasIndexHandle;
	u32 emitterIndex;
	// scales only what camera rays see directly; lighting is unchanged
	f32 backgroundIntensity = 1.0f;
};

struct WavefrontGroundHeating {
	bool enabled;
	f32 groundHeight;
	f32 hazeHeight;
	f32 gradient;
	f32 noiseAmplitude;
	f32 noiseScale;
};

struct WavefrontGlobalInfo {
	WavefrontCamera camera;
	WavefrontBlackhole blackhole;
	WavefrontAmbientMedium ambientMedium;
	u64 materialMediumTableVa;
	u64 materialMediumRecordsVa;
	u32 materialMediumModelCount;
	u64 sceneEmitterLightsVa;
	u32 sceneEmitterLightCount;
	bool meshNeeEnabled;
	bool envNeeEnabled;
	WavefrontEnvironment environment;
	WavefrontGroundHeating groundHeating;
	RegirSettings regir;
	u64 modelsVa;
	f32 shutterOpen;
	f32 shutterDuration;
	// meters; 0 is a pinhole
	f32 lensRadius = 0.0f;
	// meters along forward to the plane in focus
	f32 focalDistance = 1.0f;
	u32 frameIndex;
	u32 sampleIndex;
	u32 width;
	u32 height;
	// the pixels passes walk; the whole image when empty
	std::optional<WavefrontRegion> region;
	bool debugPath;
	u32v2 debugPixel;
	WavefrontCamera debugCamera;
};

// a pass keeps the sample and regir fill index it started with
void wavefront_global_update(
	Wavefront & wavefront,
	Regir const & regir,
	WavefrontGlobalInfo const & info
);

// dispatches in one pass: camera, trace + shade + nee per bounce, output
constexpr u32 skWavefrontPassStepCount = 26u;

// drops the pass in flight; the next integrate starts a new one
void wavefront_pass_restart(Wavefront & wavefront);

// the primary hit's position (w = 1 on a surface), shading normal and albedo
struct WavefrontAov {
	vkof::Image positionImage;
	u32 positionHandle;
	vkof::Image normalImage;
	u32 normalHandle;
	vkof::Image albedoImage;
	u32 albedoHandle;
};

// DLSS ray reconstruction guides at the primary hit, written with the aovs
struct WavefrontRrGuide {
	vkof::Image diffuseAlbedoImage;
	u32 diffuseAlbedoHandle;
	vkof::Image specularAlbedoImage;
	u32 specularAlbedoHandle;
	vkof::Image roughnessImage;
	u32 roughnessHandle;
	vkof::Image specularHitDistanceImage;
	u32 specularHitDistanceHandle;
	// the surface the guides follow, which smooth glass can see through
	vkof::Image positionImage;
	u32 positionHandle;
	vkof::Image normalImage;
	u32 normalHandle;
};

struct WavefrontIntegrateInfo {
	vkof::TransientImage outputImage;
	u32 outputHandle;
	// written by the pass's first shade step when set
	std::optional<WavefrontAov> aov;
	// needs aov
	std::optional<WavefrontRrGuide> rrGuide;
	// one sub-pixel offset in [0, 1) for every pixel of the pass, instead of
	// a random one per sample
	std::optional<f32v2> globalJitter;
	bool spectral;
	bool debugPath;
	// <= 0 disables, see wavefront-output.slang
	f32 fireflyClampLuminance = 0.0f;
	u32 width;
	u32 height;
	// most steps to dispatch, at least 1
	u32 stepBudget;
};

struct WavefrontIntegrateResult {
	vkof::RenderNode node;
	u32 stepCount;
	// the output image holds the finished pass
	bool passComplete;
	// region sweeps the finished pass completed, 0 while in flight
	u32 sweepCount;
};

// continues the pass in flight, or starts one
[[nodiscard]] WavefrontIntegrateResult wavefront_integrate(
	Wavefront & wavefront,
	WavefrontIntegrateInfo const & info
);

struct WavefrontAccumulateInfo {
	vkof::TransientImage sampleImage;
	u32 sampleHandle;
	vkof::Image resolvedImage;
	u32 resolvedHandle;
	vkof::TransientImage outputImage;
	u32 outputHandle;
	bool addSample;
	bool reset;
	f32 exposure;
	u32 width;
	u32 height;
};

struct DebugPath;

void wavefront_debug_path_prepare(Wavefront & wavefront);

void wavefront_debug_path_read(Wavefront const & wavefront, DebugPath & path);

// a reset overwrites each pixel at its next sample, over the next sweep
[[nodiscard]] vkof::RenderNode wavefront_accumulate(
	Wavefront & wavefront,
	WavefrontAccumulateInfo const & info
);

} // namespace ponder
