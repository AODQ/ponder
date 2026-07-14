// environment background sampling; dir is y-up world space

#define ENV_MODE_FURNACE 0
#define ENV_MODE_CHECKERBOARD 1
#define ENV_MODE_BLACK 2

vec3 sampleEnvironment(
	const vec3 dir,
	const float envIntensity,
	const int envMode
) {
	if (envMode == ENV_MODE_CHECKERBOARD) {
		const float checkerSize = 0.1f;
		const float checker = (
			mod(floor(dir.x / checkerSize) + floor(dir.z / checkerSize), 2.0f)
		);
		return mix(vec3(0.2f), vec3(1.0f), checker) * envIntensity;
	}
	if (envMode == ENV_MODE_BLACK) {
		return vec3(0.0f);
	}
	// ENV_MODE_FURNACE
	return f32v3(1.0f) * envIntensity;
}
