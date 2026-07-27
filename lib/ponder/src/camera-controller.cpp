#include <ponder/camera-controller.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// -----------------------------------------------------------------------------
// -- private api
// -----------------------------------------------------------------------------

static f32 sScrollDelta = 0.0f;

static void scroll_callback(GLFWwindow * const w, double const x, double const y) {
	ImGui_ImplGlfw_ScrollCallback(w, x, y);
	sScrollDelta += (f32)y;
}

// -----------------------------------------------------------------------------
// -- public api
// -----------------------------------------------------------------------------

ponder::CameraController ponder::camera_controller_create(
	srat::CameraOrbit const & orbit
) {
	CameraController c;
	c.orbit = orbit;
	camera_controller_sync_fp_from_orbit(c);
	return c;
}

void ponder::camera_controller_sync_fp_from_orbit(CameraController & c) {
	c.firstPerson.position = srat::camera_orbit_eye(c.orbit);
	c.firstPerson.yaw = -c.orbit.azimuth;
	c.firstPerson.pitch = -c.orbit.elevation;
	c.firstPerson.fovY = c.orbit.fovY;
	c.firstPerson.aspect = c.orbit.aspect;
	c.firstPerson.near = c.orbit.near;
	c.firstPerson.far = c.orbit.far;
}

void ponder::camera_controller_sync_orbit_from_fp(CameraController & c) {
	c.orbit.target = (
		c.firstPerson.position
		+ srat::camera_fp_forward(c.firstPerson) * c.orbit.distance
	);
	c.orbit.azimuth = -c.firstPerson.yaw;
	c.orbit.elevation = -c.firstPerson.pitch;
	c.orbit.fovY = c.firstPerson.fovY;
}

void ponder::camera_controller_install_scroll_callback(
	GLFWwindow * const window
) {
	glfwSetScrollCallback(window, scroll_callback);
}

bool ponder::camera_controller_update(
	CameraController & c,
	GLFWwindow * const window,
	f32 const boundsExtent,
	f32 const deltaTime
) {
	f64 curMouseX, curMouseY;
	glfwGetCursorPos(window, &curMouseX, &curMouseY);
	if (!c.mouseInitialized) {
		c.prevMouseX = curMouseX;
		c.prevMouseY = curMouseY;
		c.mouseInitialized = true;
	}
	f32 const dx = (f32)(curMouseX - c.prevMouseX);
	f32 const dy = (f32)(curMouseY - c.prevMouseY);
	c.prevMouseX = curMouseX;
	c.prevMouseY = curMouseY;

	bool cameraMoved = false;
	bool const mouseFree = !ImGui::GetIO().WantCaptureMouse;
	bool const keysFree = !ImGui::GetIO().WantCaptureKeyboard;

	bool const tabDown = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
	if (keysFree && tabDown && !c.tabWasDown) {
		c.isFirstPerson = !c.isFirstPerson;
		if (c.isFirstPerson) {
			camera_controller_sync_fp_from_orbit(c);
		} else {
			camera_controller_sync_orbit_from_fp(c);
		}
	}
	c.tabWasDown = tabDown;

	if (
		mouseFree
		&& glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS
		&& (dx != 0.0f || dy != 0.0f)
	) {
		if (c.isFirstPerson) {
			srat::camera_fp_look(c.firstPerson, dx * 0.005f, -dy * 0.005f);
		} else {
			srat::camera_orbit_rotate(c.orbit, -dx * 0.005f, dy * 0.005f);
		}
		cameraMoved = true;
	}
	if (
		!c.isFirstPerson
		&& mouseFree
		&& glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS
		&& (dx != 0.0f || dy != 0.0f)
	) {
		srat::camera_orbit_pan(
			c.orbit, -dx * 0.002f * c.orbit.distance, dy * 0.002f * c.orbit.distance
		);
		cameraMoved = true;
	}
	if (c.isFirstPerson && keysFree) {
		f32v3 move { 0.0f, 0.0f, 0.0f };
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { move.z += 1.0f; }
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { move.z -= 1.0f; }
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { move.x += 1.0f; }
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { move.x -= 1.0f; }
		if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { move.y += 1.0f; }
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { move.y -= 1.0f; }
		if (move.x != 0.0f || move.y != 0.0f || move.z != 0.0f) {
			f32 const speed = boundsExtent * c.flySpeedMultiplier * deltaTime;
			srat::camera_fp_move(c.firstPerson, move * speed);
			cameraMoved = true;
		}
	}
	if (mouseFree && sScrollDelta != 0.0f) {
		if (c.isFirstPerson) {
			c.flySpeedMultiplier *= expf(sScrollDelta * 0.1f);
		} else {
			srat::camera_orbit_zoom(c.orbit, -sScrollDelta * 0.1f * c.orbit.distance);
			cameraMoved = true;
		}
	}
	sScrollDelta = 0.0f;

	return cameraMoved;
}
