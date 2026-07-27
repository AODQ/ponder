#pragma once

#include <srat/camera.hpp>

struct GLFWwindow;

namespace ponder {

struct CameraController {
	srat::CameraOrbit orbit {};
	srat::CameraFirstPerson firstPerson {};
	bool isFirstPerson = false;
	f32 flySpeedMultiplier = 0.5f;

	// frame-to-frame input state; camera_controller_update owns these
	f64 prevMouseX = 0.0;
	f64 prevMouseY = 0.0;
	bool mouseInitialized = false;
	bool tabWasDown = false;
};

[[nodiscard]] CameraController camera_controller_create(
	srat::CameraOrbit const & orbit
);

// keeps toggling between orbit/first-person from jumping the view
void camera_controller_sync_fp_from_orbit(CameraController & c);
void camera_controller_sync_orbit_from_fp(CameraController & c);

// chains into whatever glfw scroll handler is already installed (imgui's)
// instead of replacing it; call once after the window exists
void camera_controller_install_scroll_callback(GLFWwindow * const window);

// returns true if the camera moved this frame
bool camera_controller_update(
	CameraController & c,
	GLFWwindow * const window,
	f32 const boundsExtent,
	f32 const deltaTime
);

} // namespace ponder
