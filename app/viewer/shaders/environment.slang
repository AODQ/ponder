module "environment";
__exported import "util-types";
__exported import "util-environment-map";
__exported import "util-mis";
// (TODO REVIEW)
// (TODO REVIEW)
// environment background sampling; dir is y-up world space


public static const u32 ENV_MODE_FURNACE = 0u;
public static const u32 ENV_MODE_CHECKERBOARD = 1u;
public static const u32 ENV_MODE_BLACK = 2u;
public static const u32 ENV_MODE_HDRMAP = 3u;
public func sampleEnvironment(
	dir: f32v3,
	envIntensity: f32,
	envMode: i32,
	envHandles: EnvironmentMapHandles,
	envRotation: f32,
) -> f32v3 {
	if (envMode == ENV_MODE_HDRMAP) {
		return (
			environmentMapRadiance(envHandles, dir, envRotation) * envIntensity
		);
	}
	if (envMode == ENV_MODE_CHECKERBOARD) {
		const f32 checkerSize = 0.1f;
		const f32 checker = (
			mod(floor(dir.x / checkerSize) + floor(dir.z / checkerSize), 2.0f)
		);
		return lerp(f32v3(0.2f), f32v3(1.0f), checker) * envIntensity;
	}
	if (envMode == ENV_MODE_BLACK) {
		return f32v3(0.0f);
	}
	// ENV_MODE_FURNACE
	return f32v3(1.0f) * envIntensity;
}

// (TODO REVIEW)
