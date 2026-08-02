implementing types;

public typealias u32 = uint32_t;
public typealias u32v2 = uint32_t2;
public typealias u32v3 = uint32_t3;
public typealias u32v4 = uint32_t4;

public typealias u64 = uint64_t;

public typealias i32 = int32_t;
public typealias i32v2 = int32_t2;
public typealias i32v3 = int32_t3;
public typealias i32v4 = int32_t4;

public typealias f32 = float32_t;
public typealias f32v2 = float32_t2;
public typealias f32v3 = float32_t3;
public typealias f32v4 = float32_t4;

public typealias f32m3x3 = float32_t3x3;
public typealias f32m4x4 = float32_t4x4;

public typealias f32v3x2 = float32_t2x3;

// -----------------------------------------------------------------------------

[[vk::binding(0, 0)]] public Sampler2D vkofTextures[];
[[vk::binding(1, 0)]] public RWTexture2D<u32> vkofImagesU[];
[[vk::binding(2, 0)]] public RWTexture2D<f32v4> vkofImages[];
[[vk::binding(3, 0)]] public RaytracingAccelerationStructure vkofTlas;
[[vk::binding(4, 0)]] public Sampler3D vkofTextures3D[];
[[vk::binding(5, 0)]] public RWTexture3D<f32v4> vkofImages3D[];

// -----------------------------------------------------------------------------

public struct PushConstantGlobal {
	public f32 * rgbToSpectrum;
	public u32 rgbToSpectrumResolution;
	public u32 frameIndex;
	public void * models;
	public u32 renderWidth;
	public u32 renderHeight;
	public f32v3 cameraOri;
	public f32 cameraShutterOpen;
	public f32v3 cameraForward;
	public f32 cameraShutterDuration;
	// scaled by tan(fovY / 2) * aspect
	public f32v3 cameraRight;
	// farmes since the last accumulation reset
	public u32 sampleIndex;
	// scaled by tan(fovY / 2)
	public f32v3 cameraUp;
	// gravitational radius G M / c^2 in meters; 0 disables the hole
	public f32 blackholeMass;
	public f32v3 blackholeOri;
	// lens sphere radius as a multiple of blackholeMass
	public f32 blackholeLensRadius;
	public u32 blackholeMaxChords;
	public f32 blackholeDiskInnerRadius;
	public f32 blackholeDiskOuterRadius;
	// peak disk temperature in kelvin
	public f32 blackholeDiskTemperature;
	// display-white at 555 nm
	public f32 blackholeDiskBrightness;
	// optical depth at the inner radius, falling as (inner / r)^exponent
	public f32 blackholeDiskOpticalDepth;
	public f32 blackholeDiskOpticalDepthExponent;
	// rk4 step length as a fraction of the distance to the blackhole
	public f32 blackholeStepFactor;
	// disk animation time in meters of light travel
	public f32 blackholeDiskTime;
	// +1 or -1, flips which side of the disk approaches the camera
	public f32 blackholeDiskRotation;
	public f32 blackholeDiskNoiseAmplitude;
	// noise features per unit of log radius and per radian
	public f32 blackholeDiskNoiseScale;
	// orbiting hot spot, radius and size as multiples of blackholeMass
	public f32 blackholeDiskHotspotRadius;
	public f32 blackholeDiskHotspotTemperature;
	public f32 blackholeDiskHotspotSize;
	// a / M, signed, |spin| < 1
	public f32 blackholeSpin;
	// null when no pixel is selected
	public vkof::DebugPath * debugPath;
	public u32v2 debugPixel;
	// view the debug pixel was picked from, scaled like camera*
	public f32v3 debugCameraOri;
	u32 pad1;
	public f32v3 debugCameraForward;
	u32 pad2;
	public f32v3 debugCameraRight;
	u32 pad3;
	public f32v3 debugCameraUp;
	u32 pad4;
	// scene-wide medium, see WavefrontAmbientMedium; σ per meter
	public f32v3 ambientMediumSigmaAbsorption;
	// vkof::MediumType
	public u32 ambientMediumType;
	public f32v3 ambientMediumSigmaScattering;
	// vkof::PhaseFunctionType
	public u32 ambientMediumPhase;
	public f32 ambientMediumAnisotropy;
	public f32 ambientMediumDraineAlpha;
	public f32 ambientMediumDropletDiameter;
	// height fog density is
	// σ·exp(-(y - ambientMediumHeightOrigin) / ambientMediumHeightFalloff)
	public f32 ambientMediumHeightOrigin;
	public f32 ambientMediumHeightFalloff;
	u32 pad5;
	u32 pad6;
	u32 pad7;
	// u32v2 { first record, record count } per model, 0 when unused
	public u32v2 * materialMediumTable;
	public vkof::MaterialMediumRecord * materialMediumRecords;
	public u32 materialMediumModelCount;
	public u32 sceneEmitterLightCount;
	public vkof::SceneEmitterLight * sceneEmitterLights;
	public u32 meshNeeEnabled;
	public u32 envNeeEnabled;
	// vkof::EnvironmentMode
	public u32 environmentMode;
	public f32 environmentIntensity;
	// radians about +y
	public f32 environmentRotation;
	// texture handles of the hdr map and its importance-sampling tables
	public u32 environmentRadianceHandle;
	public u32 environmentPdfHandle;
	public u32 environmentAliasProbHandle;
	public u32 environmentAliasIndexHandle;
	// the environment entry of sceneEmitterLights; none when out of range
	public u32 environmentEmitterIndex;
	// ground-heating index field: a layer above groundHeight, hazeHeight thick,
	// where the refractive index rises by gradient per unit of height
	public f32 groundHeatingGroundHeight;
	public f32 groundHeatingHazeHeight;
	public f32 groundHeatingGradient;
	public f32 groundHeatingNoiseAmplitude;
	public f32 groundHeatingNoiseScale;
	// 0 bypasses the index field, rays stay straight
	public u32 groundHeatingEnabled;

	// -- regir
	public u32 * regirGuards;
	public vkof::RegirCell * regirCells;
	public vkof::RegirReservoir * regirReservoirs;
	public u32 regirCellCount;
	public f32 regirCellSize;
	public u32 regirEnabled;
	public u32 regirFillIndex;
	// frames to evict regir cell
	public u32 regirMaxAge;
	// 0 keeps each cell's representative, see regirRecordShadingPoint
	public u32 regirRefresh;
	// nonzero: nee looks up a random neighbouring cell, see regirJitteredPosition
	public u32 regirJitter;
	u32 regirPad0;

	// -- wavefront
	// WavefrontTransportState per path
	public void * wavefrontTransportStates;
	// PathState per path
	public void * wavefrontPathStates;
	// PathHitRecord per path, from trace to shade
	public void * wavefrontHitRecords;
	// PathNeeRecord per path
	public void * wavefrontNeeRecords;

	// -- material tables
	// zeltner 2022 sheen ltc (a, b, R), 32x32 over (μ, roughness)
	public u32 zeltnerLtcParamHandle;

	// -- pass path data
	// path i takes slot passSlotFirst + i, below passSlotCount
	// row-major 16x16 tiles
	public u32 passSlotFirst;
	public u32 passSlotCount;
	public u32 passPathCount;
	public u32v2 passRegionOrigin;
	public u32v2 passRegionExtent;
	public u32 passRegionTilesW;

	// scales the environment seen directly by camera rays only
	public f32 environmentBackgroundIntensity;

	// meters; 0 is a pinhole
	public f32 cameraLensRadius;
	// meters along forward to the plane in focus
	public f32 cameraFocalDistance;
};

namespace vkof {

public struct PushConstants<T> {
	[[vk::offset(104)]] public PushConstantGlobal * global;
	[[vk::offset(128)]] public T stage;
};

}


// -----------------------------------------------------------------------------

static PushConstantGlobal * sImplGlobal;

namespace vkof {

public func globalBind(global: PushConstantGlobal *) {
	::sImplGlobal = global;
}

public property PushConstantGlobal * global {
	get { return ::sImplGlobal; }
}

}
