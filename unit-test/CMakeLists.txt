# ---------------------------------------------------------------------------
# doctest
# ---------------------------------------------------------------------------

FetchContent_Declare(
	doctest
	GIT_REPOSITORY https://github.com/doctest/doctest.git
	GIT_TAG		v2.4.11
	GIT_SHALLOW	TRUE
)
FetchContent_MakeAvailable(doctest)

# ---------------------------------------------------------------------------
# vkof-test executable
# ---------------------------------------------------------------------------

add_executable(
	vkof-test
	src/main.cpp
	src/test-buffer.cpp
	src/test-accel.cpp
	src/test-buffer.cpp
	src/test-compute.cpp
	src/test-pipeline.cpp
	src/test-rendernode.cpp
	src/test-sampler.cpp
	src/test-image.cpp
	src/test-transient.cpp
	src/test-noise.cpp
	src/test-fresnel.cpp
	src/test-microfacet.cpp
	src/test-shading-frame.cpp
	src/test-roughness.cpp
	src/test-energy-compensation.cpp
	src/test-dielectric-specular.cpp
	src/test-eon-diffuse.cpp
	src/test-glossy-diffuse-layer.cpp
	src/test-metallic-lobe.cpp
	src/test-openpbr-ibl.cpp
	src/test-microfacet-sampling.cpp
	src/test-diffuse-sampling.cpp
	src/test-fuzz-sampling.cpp
	src/test-fuzz-eval.cpp
	src/test-diffuse-transmission.cpp
	src/test-transmission.cpp
	src/test-transmission-sampling.cpp
	src/test-mixture-lobe-selection.cpp
	src/test-dispatch-readback-race.cpp
	src/test-shading-single-bounce.cpp
	src/test-thinfilm.cpp
	src/test-vkof-validation.cpp
	src/test-furnace-model-render.cpp
	src/test-environment-sampling.cpp
	src/test-vdb-sampling.cpp
	src/test-vdb-phase.cpp
	src/test-subsurface-albedo.cpp
	src/test-subsurface-walk.cpp
	src/test-subsurface-entry-sampling.cpp
	src/test-subsurface-furnace.cpp
	src/test-subsurface-banding-diagnostic.cpp
	src/test-thin-walled-subsurface.cpp
	src/test-stage.cpp
	src/test-blackbody.cpp
	src/test-tangent-generation.cpp
)

# Bake absolute paths into the binary
target_compile_definitions(
	vkof-test
	PRIVATE
	TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders/"
	APP_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../app/viewer/shaders"
	PONDER_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../lib/ponder/shaders/"
	PONDER_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../lib/ponder/assets/"
	MOR_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../lib/mor/include/"
	NOISE_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	FRESNEL_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	MICROFACET_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	SHADING_FRAME_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	ENERGY_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	DIELECTRIC_SPECULAR_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	EON_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	GLOSSY_LAYER_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	METALLIC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	OPENPBR_IBL_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	SAMPLING_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	FUZZ_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	TRANSMISSION_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	CORNELL_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	MIXTURE_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	RAYCAST_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	THINFILM_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	FURNACE_MODEL_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	ENVIRONMENT_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	VDB_PHASE_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	SUBSURFACE_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/output/"
	LIB_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../lib"
	REPO_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_include_directories(
	vkof-test
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../app
	${CMAKE_CURRENT_SOURCE_DIR}/../app/shaders
)

target_include_directories(
	vkof-test
	SYSTEM PRIVATE
	${stb_SOURCE_DIR}
	${tinyexr_SOURCE_DIR}
)

target_link_libraries(
	vkof-test
	PRIVATE
	ponder-flags
	vkof
	mor
	ponder
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	glfw
	GPUOpen::VulkanMemoryAllocator
	doctest::doctest
	stage
)

target_link_options(
	vkof-test
	PRIVATE
	-fuse-ld=mold
	${san-link}
)

install(
	TARGETS vkof-test
	RUNTIME DESTINATION bin
)

set_target_properties(vkof-test PROPERTIES INSTALL_RPATH "$ORIGIN")
