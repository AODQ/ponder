#pragma once

#include <srat/core-math.hpp>

namespace srat {

// -----------------------------------------------------------------------------
// -- first person camera
// -----------------------------------------------------------------------------

struct CameraFirstPerson {
	f32v3 position;
	f32 yaw;
	f32 pitch;
	f32 fovY;
	f32 aspect;
	f32 near;
	f32 far;
};

[[nodiscard]] f32v3 camera_fp_forward(CameraFirstPerson const & cam);
[[nodiscard]] f32v3 camera_fp_right(CameraFirstPerson const & cam);
[[nodiscard]] f32m44 camera_fp_view(CameraFirstPerson const & cam);
[[nodiscard]] f32m44 camera_fp_proj(CameraFirstPerson const & cam);

void camera_fp_look(CameraFirstPerson & cam, f32 const dx, f32 const dy);
void camera_fp_move(CameraFirstPerson & cam, f32v3 const localDelta);

// -----------------------------------------------------------------------------
// -- orbit camera
// -----------------------------------------------------------------------------

struct CameraOrbit {
	f32v3 target;
	f32 distance;
	f32 azimuth;
	f32 elevation;
	f32 fovY;
	f32 aspect;
	f32 near;
	f32 far;
};

[[nodiscard]] f32v3 camera_orbit_eye(CameraOrbit const & cam);
[[nodiscard]] f32m44 camera_orbit_view(CameraOrbit const & cam);
[[nodiscard]] f32m44 camera_orbit_proj(CameraOrbit const & cam);

void camera_orbit_rotate(CameraOrbit & cam, f32 const dAzimuth, f32 const dElevation);
void camera_orbit_zoom(CameraOrbit & cam, f32 const delta);
void camera_orbit_pan(CameraOrbit & cam, f32 const dx, f32 const dy);

// -----------------------------------------------------------------------------
// -- orthographic camera
// -----------------------------------------------------------------------------

struct CameraOrtho {
	f32v3 position;
	f32 halfWidth;
	f32 halfHeight;
	f32 near;
	f32 far;
};

[[nodiscard]] f32m44 camera_ortho_view(CameraOrtho const & cam);
[[nodiscard]] f32m44 camera_ortho_proj(CameraOrtho const & cam);

} // namespace srat
