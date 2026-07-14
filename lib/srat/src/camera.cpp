#include <srat/camera.hpp>

#include <cmath>

namespace srat {

static constexpr f32v3 skWorldUp = {0.0f, 1.0f, 0.0f};
static constexpr f32 skPitchLimit = 1.5533f;
static constexpr f32 skElevationLimit = 1.5533f;
static constexpr f32 skMinOrbitDistance = 0.001f;

// -----------------------------------------------------------------------------
// -- first person camera
// -----------------------------------------------------------------------------

f32v3 camera_fp_forward(CameraFirstPerson const & cam) {
	return {
		cosf(cam.pitch) * sinf(cam.yaw),
		sinf(cam.pitch),
		-cosf(cam.pitch) * cosf(cam.yaw),
	};
}

f32v3 camera_fp_right(CameraFirstPerson const & cam) {
	return f32v3_normalize(f32v3_cross(camera_fp_forward(cam), skWorldUp));
}

f32m44 camera_fp_view(CameraFirstPerson const & cam) {
	f32v3 const forward = camera_fp_forward(cam);
	return f32m44_lookat(cam.position, cam.position + forward, skWorldUp);
}

f32m44 camera_fp_proj(CameraFirstPerson const & cam) {
	return f32m44_perspective(cam.fovY, cam.aspect, cam.near, cam.far);
}

void camera_fp_look(CameraFirstPerson & cam, f32 const dx, f32 const dy) {
	cam.yaw += dx;
	cam.pitch = f32_clamp(cam.pitch + dy, -skPitchLimit, skPitchLimit);
}

void camera_fp_move(CameraFirstPerson & cam, f32v3 const localDelta) {
	f32v3 const forward = camera_fp_forward(cam);
	f32v3 const right = camera_fp_right(cam);
	cam.position = (
		cam.position
		+ right * localDelta.x
		+ skWorldUp * localDelta.y
		+ forward * localDelta.z
	);
}

// -----------------------------------------------------------------------------
// -- orbit camera
// -----------------------------------------------------------------------------

f32v3 camera_orbit_eye(CameraOrbit const & cam) {
	return {
		cam.target.x + cam.distance * cosf(cam.elevation) * sinf(cam.azimuth),
		cam.target.y + cam.distance * sinf(cam.elevation),
		cam.target.z + cam.distance * cosf(cam.elevation) * cosf(cam.azimuth),
	};
}

f32m44 camera_orbit_view(CameraOrbit const & cam) {
	return f32m44_lookat(camera_orbit_eye(cam), cam.target, skWorldUp);
}

f32m44 camera_orbit_proj(CameraOrbit const & cam) {
	return f32m44_perspective(cam.fovY, cam.aspect, cam.near, cam.far);
}

void camera_orbit_rotate(CameraOrbit & cam, f32 const dAzimuth, f32 const dElevation) {
	cam.azimuth += dAzimuth;
	cam.elevation = f32_clamp(cam.elevation + dElevation, -skElevationLimit, skElevationLimit);
}

void camera_orbit_zoom(CameraOrbit & cam, f32 const delta) {
	cam.distance = f32_max(skMinOrbitDistance, cam.distance + delta);
}

void camera_orbit_pan(CameraOrbit & cam, f32 const dx, f32 const dy) {
	f32v3 const eye = camera_orbit_eye(cam);
	f32v3 const forward = f32v3_normalize(cam.target - eye);
	f32v3 const right = f32v3_normalize(f32v3_cross(forward, skWorldUp));
	f32v3 const up = f32v3_cross(right, forward);
	cam.target = (
		cam.target
		+ right * dx
		+ up * dy
	);
}

// -----------------------------------------------------------------------------
// -- orthographic camera
// -----------------------------------------------------------------------------

f32m44 camera_ortho_view(CameraOrtho const & cam) {
	return f32m44_lookat(
		cam.position,
		{cam.position.x, cam.position.y, cam.position.z - 1.0f},
		skWorldUp
	);
}

f32m44 camera_ortho_proj(CameraOrtho const & cam) {
	f32 const rw = 1.0f / cam.halfWidth;
	f32 const rh = 1.0f / cam.halfHeight;
	f32 const d = cam.far - cam.near;
	return {
		rw, 0.0f, 0.0f, 0.0f,
		0.0f, -rh, 0.0f, 0.0f,
		0.0f, 0.0f, -2.0f / d, 0.0f,
		0.0f, 0.0f, -(cam.far + cam.near) / d, 1.0f,
	};
}

} // namespace srat
