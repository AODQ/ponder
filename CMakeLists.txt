cmake_minimum_required(VERSION 3.20)
project(ponder)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# CMake 4+ removed support for cmake_minimum_required versions < 3.5.
# Allow FetchContent'd projects that still declare VERSION 3.x to configure.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# ------------------------------------------------------------------------------
# -- ccache
# ------------------------------------------------------------------------------

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK	ccache)
endif()

# ------------------------------------------------------------------------------
# -- sanitizers
# ------------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL Debug)
	set(san-flags -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
	set(san-link  -fsanitize=address -fsanitize=undefined)
else()
	set(san-flags "")
	set(ssan-link "")
endif()

# ------------------------------------------------------------------------------
# -- dependencies (FetchContent)
# ------------------------------------------------------------------------------

include(FetchContent)
# check for latest tags at:
#   https://github.com/charles-lunarg/vk-bootstrap/releases
#   https://github.com/glfw/glfw/releases
#   https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/releases

FetchContent_Declare(
	vk-bootstrap
	GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
	GIT_TAG		v1.4.350
	GIT_SHALLOW	TRUE
)

FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw.git
	GIT_TAG		3.4
	GIT_SHALLOW	TRUE
)

FetchContent_Declare(
	vma
	GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
	GIT_TAG		v3.1.0
	GIT_SHALLOW	TRUE
)

set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(vk-bootstrap glfw vma)
FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui.git
	GIT_TAG docking
	GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
	imguizmo
	GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git
	GIT_TAG master
	GIT_SHALLOW TRUE
)
FetchContent_GetProperties(imguizmo)
if(NOT imguizmo_POPULATED)
	FetchContent_Populate(imguizmo)
endif()

FetchContent_Declare(
	cgltf
	GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
	GIT_TAG        v1.15
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cgltf)

# system openvdb; its find module hard-requires boost headers that the
# library itself doesn't need at consume time, so link it directly
add_library(openvdb-system SHARED IMPORTED)
set_target_properties(
	openvdb-system
	PROPERTIES
	IMPORTED_LOCATION /usr/lib/libopenvdb.so
	# openvdb headers inline references into these (imath's half<->float
	# table, tbb templates), so consumers must link them explicitly
	INTERFACE_LINK_LIBRARIES "/usr/lib/libImath.so;/usr/lib/libtbb.so"
)

FetchContent_Declare(
	meshoptimizer
	GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
	GIT_TAG        v0.21
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(meshoptimizer)

FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG        master
	GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

FetchContent_Declare(
	tinyexr
	GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
	GIT_TAG        release
	GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(tinyexr)
if(NOT tinyexr_POPULATED)
	FetchContent_Populate(tinyexr)
endif()

FetchContent_Declare(
	rapidjson
	GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
	GIT_TAG        master
	GIT_SHALLOW    TRUE
)
set(RAPIDJSON_BUILD_DOC      OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(rapidjson)

# ------------------------------------------------------------------------------
# -- Vulkan
# ------------------------------------------------------------------------------

find_package(Vulkan REQUIRED)

# ------------------------------------------------------------------------------
# -- NVIDIA Aftermath
# ------------------------------------------------------------------------------

option(VKOF_AFTERMATH "Enable NVIDIA Aftermath GPU crash dumps" ON)
if(VKOF_AFTERMATH)
	list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
	find_package(Aftermath REQUIRED)
endif()

# ------------------------------------------------------------------------------
# -- NVIDIA DLSS (ray reconstruction)
# ------------------------------------------------------------------------------

# proprietary sdk (NVIDIA RTX SDKs license), fetched only when enabled
option(PONDER_DLSS "Enable the DLSS ray reconstruction realtime mode" OFF)
if(PONDER_DLSS)
	FetchContent_Declare(
		dlss
		GIT_REPOSITORY https://github.com/NVIDIA/DLSS.git
		GIT_TAG        v310.9.16
		GIT_SHALLOW    TRUE
	)
	FetchContent_GetProperties(dlss)
	if(NOT dlss_POPULATED)
		FetchContent_Populate(dlss)
	endif()

	add_library(dlss-ngx STATIC IMPORTED)
	set_target_properties(
		dlss-ngx
		PROPERTIES
		IMPORTED_LOCATION "${dlss_SOURCE_DIR}/lib/Linux_x86_64/libnvsdk_ngx.a"
		INTERFACE_INCLUDE_DIRECTORIES "${dlss_SOURCE_DIR}/include"
		INTERFACE_LINK_LIBRARIES "dl;stdc++"
	)
	set(dlss-rr-library
		"${dlss_SOURCE_DIR}/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.310.9.16"
	)
endif()

# ------------------------------------------------------------------------------
# -- interface compile flags
# ------------------------------------------------------------------------------

add_library(ponder-flags INTERFACE)

target_compile_options(
	ponder-flags
	INTERFACE
	-Wextra
	-Wpedantic
	$<$<CONFIG:Debug>:-O0 -g3 -ggdb>
	$<$<CONFIG:RelWithDebInfo>:-O2 -g3 -ggdb>
	$<$<CONFIG:Release>:-O3>
	${san-flags}
)

# ------------------------------------------------------------------------------
# -- generated slang views of the shared c++/glsl headers
# ------------------------------------------------------------------------------

# slang cannot include the `#ifdef __cplusplus` headers: it has no
# layout(buffer_reference) and no layout(offset=). the shared structs are
# regenerated here instead so the two views cannot drift, since a field
# reordered within the same size scrambles values with no error.
# see plan-slang-port.md §6

set(slang-shared-headers
	lib/mor/include/mor/mor-shared.h            util-mor-shared
	lib/ponder/shaders-old/emitter-shared.h     util-emitter-shared
	lib/ponder/shaders-old/util-regir-shared.h  util-regir-shared
	app/viewer/shaders/shared/global_pc.h       util-global-pc
	app/viewer/shaders/resolve_pc.h             util-resolve-pc
	app/viewer/shaders/bloom-blur-pc.h          util-bloom-blur-pc
	app/viewer/shaders/bloom-composite-pc.h     util-bloom-composite-pc
	app/viewer/shaders/bloom-extract-pc.h       util-bloom-extract-pc
	app/viewer/shaders/display-dither-pc.h      util-display-dither-pc
	app/viewer/shaders/nan-check-pc.h           util-nan-check-pc
	app/viewer/shaders/pt-accumulate-pc.h       util-pt-accumulate-pc
	app/viewer/shaders/pt-denoise-pc.h          util-pt-denoise-pc
	app/viewer/shaders/manifold-shade-pc.h      util-manifold-shade-pc
	app/viewer/shaders/regir-pc.h               util-regir-pc
	app/observer/shaders/scene-visibility-resolve-pc.h util-scene-visibility-resolve-pc
	app/observer/shaders/shared/global-pc.h     util-observer-global-pc
	lib/ponder/shaders-old/env-prefilter-pc.h   util-env-prefilter-pc
)

set(slang-generated-outputs)
list(LENGTH slang-shared-headers slang-shared-count)
math(EXPR slang-shared-last "${slang-shared-count} / 2 - 1")
foreach(i RANGE ${slang-shared-last})
	math(EXPR pathIndex "${i} * 2")
	math(EXPR nameIndex "${i} * 2 + 1")
	list(GET slang-shared-headers ${pathIndex} headerPath)
	list(GET slang-shared-headers ${nameIndex} moduleName)
	set(outPath "${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders-old/${moduleName}.slang")
	add_custom_command(
		OUTPUT ${outPath}
		COMMAND ${CMAKE_COMMAND} -E make_directory
			"${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders-old/generated"
		COMMAND python3
			"${CMAKE_CURRENT_SOURCE_DIR}/lib/tools/shared-header-to-slang.py"
			"${CMAKE_CURRENT_SOURCE_DIR}/${headerPath}"
			"${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders-old/generated/${moduleName}.slang"
			--module ${moduleName}
			--macros-dir "${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders-old"
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders-old/generated/${moduleName}.slang"
			${outPath}
		DEPENDS
			"${CMAKE_CURRENT_SOURCE_DIR}/${headerPath}"
			"${CMAKE_CURRENT_SOURCE_DIR}/lib/tools/shared-header-to-slang.py"
		COMMENT "slang view of ${headerPath}"
		VERBATIM
	)
	list(APPEND slang-generated-outputs ${outPath})
endforeach()

add_custom_target(slang-shared-headers ALL DEPENDS ${slang-generated-outputs})

# ------------------------------------------------------------------------------
# -- srat (non-graphics utility library)
# ------------------------------------------------------------------------------

add_library(
	srat
	STATIC
	lib/srat/src/alloc-virtual-range.cpp
	lib/srat/src/camera.cpp
)

target_include_directories(
	srat
	PUBLIC
	lib/srat/include
)

target_link_libraries(
	srat
	PRIVATE
	ponder-flags
)

# ------------------------------------------------------------------------------
# -- vkof (vuck off)
# ------------------------------------------------------------------------------

add_library(
	vkof
	STATIC
	lib/vkof/src/vkof.cpp
	lib/vkof/src/shader-cache.cpp
	lib/vkof/src/vma.cpp
	lib/vkof/src/stb_image_write.cpp
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

if(VKOF_AFTERMATH)
	target_sources(vkof PRIVATE lib/vkof/src/aftermath.cpp)
	target_compile_definitions(vkof PRIVATE VKOF_AFTERMATH)
	target_link_libraries(vkof PRIVATE Aftermath::Aftermath)
	install(FILES "${Aftermath_LIBRARY}" DESTINATION bin)
endif()

target_link_libraries(
	vkof
	PUBLIC
	srat
	PRIVATE
	ponder-flags
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	glfw
	GPUOpen::VulkanMemoryAllocator
)

target_include_directories(
	vkof
	PUBLIC
	lib/vkof/include
)

if(PONDER_DLSS)
	target_sources(vkof PRIVATE lib/vkof/src/ngx.cpp)
	target_compile_definitions(vkof PUBLIC PONDER_DLSS=1)
	target_link_libraries(vkof PRIVATE dlss-ngx)
	set_source_files_properties(
		lib/vkof/src/ngx.cpp
		PROPERTIES SKIP_PRECOMPILE_HEADERS ON
	)
endif()

target_include_directories(
	vkof
	SYSTEM PUBLIC
	${imgui_SOURCE_DIR}
	PRIVATE
	${vma_SOURCE_DIR}/include
	${stb_SOURCE_DIR}
	${tinyexr_SOURCE_DIR}
	${rapidjson_SOURCE_DIR}/include
	# vkof.cpp/aftermath.cpp #include "aftermath.hpp" unqualified
	lib/vkof/include/vkof
)

target_precompile_headers(
	vkof
	PRIVATE
	<VkBootstrap.h>
	<GLFW/glfw3.h>
)

set_source_files_properties(
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
	PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)

# ------------------------------------------------------------------------------
# -- mor (glTF scene loading + meshlet building)
# ------------------------------------------------------------------------------

add_library(
	mor
	STATIC
	lib/mor/src/mor.cpp
	lib/mor/src/stb_image.cpp
)

target_include_directories(
	mor
	PUBLIC
	lib/mor/include
)
target_include_directories(
	mor
	SYSTEM PRIVATE ${cgltf_SOURCE_DIR} ${stb_SOURCE_DIR}
)
target_link_libraries(
	mor
	PUBLIC
	vkof srat
	PRIVATE
	ponder-flags
	meshoptimizer
)

# ------------------------------------------------------------------------------
# -- ponder (app-specific implementation: openpbr, scene, etc.)
# ------------------------------------------------------------------------------

add_library(
	ponder
	STATIC
	lib/ponder/src/vdb.cpp
	lib/ponder/src/bluenoise.cpp
	lib/ponder/src/energy-tables.cpp
	lib/ponder/src/zeltner-tables.cpp
	lib/ponder/src/stb-image-impl.cpp
	lib/ponder/src/tinyexr-impl.cpp
	lib/ponder/src/environment-tables.cpp
	lib/ponder/src/camera-controller.cpp
	lib/ponder/src/asset-browser.cpp
	lib/ponder/src/material-inspector.cpp
	lib/ponder/src/environment-ibl.cpp
	lib/ponder/src/emitter-tables.cpp
	lib/ponder/src/wavefront.cpp
	lib/ponder/src/dlss-rr.cpp
	lib/ponder/src/timeline.cpp
	lib/ponder/src/debug-path.cpp
	lib/ponder/src/regir.cpp
	lib/ponder/src/regir-debug.cpp
	lib/ponder/src/material-media.cpp
)

target_include_directories(
	ponder
	PUBLIC
	lib/ponder/include
)

# public because emitter-shared.h's GpuEmitterLight is a dual c++/glsl struct
# consumers build the light list out of, the same way mor-shared.h works
target_include_directories(
	ponder
	PUBLIC
	lib/ponder/shaders-old
)

target_include_directories(
	ponder
	SYSTEM PUBLIC
	${imgui_SOURCE_DIR}
	PRIVATE
	${stb_SOURCE_DIR}
	${tinyexr_SOURCE_DIR}
	${rapidjson_SOURCE_DIR}/include
)

target_link_libraries(
	ponder
	PUBLIC
	vkof
	mor
	PRIVATE
	ponder-flags
	openvdb-system
	glfw
)

# nanovdb is header-only; this unlocks its openvdb-conversion code paths
target_compile_definitions(ponder PRIVATE NANOVDB_USE_OPENVDB)

# ponder's own shaders' runtime compile dir (vkof::pipeline_compute_create's
# includePaths, not the c++ compiler) -- environment-ibl.cpp bakes its ibl
# prefilter shaders directly rather than depending on a caller-supplied path
target_compile_definitions(
	ponder
	PRIVATE
	PONDER_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders-old/"
	PONDER_SHADER_NEW_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders/"
	PONDER_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/assets/"
	MOR_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/mor/include/"
)

# checks that every PonderDifferentiable shader function differentiates; not part
# of the default build. cmake --build <dir> --target ponder-autodiff-check
set(ponder-shader-dir "${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders")
add_custom_target(
	ponder-autodiff-check
	COMMAND
		slangc
		-target spirv
		-stage compute
		-entry main
		-matrix-layout-column-major
		-force-glsl-scalar-layout
		-warnings-as-errors all
		-DPONDER_AUTODIFF_CHECK
		-I ${ponder-shader-dir}/
		-o ${CMAKE_CURRENT_BINARY_DIR}/autodiff-check.spv
		${ponder-shader-dir}/entry/autodiff-check.slang
	COMMAND
		slangc
		-target spirv
		-stage compute
		-entry main
		-matrix-layout-column-major
		-force-glsl-scalar-layout
		-warnings-as-errors all
		-DPONDER_AUTODIFF_CHECK
		-DPONDER_SPECTRUM_RGB
		-I ${ponder-shader-dir}/
		-o ${CMAKE_CURRENT_BINARY_DIR}/autodiff-check-rgb.spv
		${ponder-shader-dir}/entry/autodiff-check.slang
	WORKING_DIRECTORY ${ponder-shader-dir}
	COMMENT "checking shader differentiability"
	VERBATIM
)

# ------------------------------------------------------------------------------
# -- stage (json scene file: gltf/vdb/alembic instances, cameras, lights, env)
# ------------------------------------------------------------------------------

add_library(
	stage
	STATIC
	lib/stage/src/stage.cpp
	lib/stage/src/stage-imgui.cpp
	${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp
)

target_include_directories(
	stage
	PUBLIC
	lib/stage/include
)

target_include_directories(
	stage
	SYSTEM PRIVATE
	${rapidjson_SOURCE_DIR}/include
	${imguizmo_SOURCE_DIR}/src
)

target_link_libraries(
	stage
	PUBLIC
	mor
	ponder
	srat
	PRIVATE
	ponder-flags
)

# ------------------------------------------------------------------------------
# -- observer (real-time renderer)
# ------------------------------------------------------------------------------

add_executable(
	observer
	app/observer/main.cpp
)

target_compile_definitions(
	observer
	PRIVATE
	PONDER_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders-old/"
	PONDER_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/assets/"
	MOR_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/mor/include/"
)

target_include_directories(
	observer
	SYSTEM PRIVATE
	${rapidjson_SOURCE_DIR}/include
)

target_link_libraries(
	observer
	PRIVATE
	ponder-flags
	vkof
	srat
	mor
	ponder
	stage
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	glfw
	GPUOpen::VulkanMemoryAllocator
)

target_link_options(
	observer
	PRIVATE
	-fuse-ld=mold
	${san-link}
)

install(
	TARGETS observer
	RUNTIME DESTINATION bin
)

set_target_properties(observer PROPERTIES INSTALL_RPATH "$ORIGIN")

# ------------------------------------------------------------------------------
# -- viewer (offline renderer)
# ------------------------------------------------------------------------------

add_executable(
	viewer
	app/viewer/main.cpp
	app/viewer/scene-state.cpp
)

target_compile_definitions(
	viewer
	PRIVATE
	PONDER_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders-old/"
	PONDER_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/assets/"
	MOR_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/mor/include/"
)

target_include_directories(
	viewer
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/app/viewer
	${rapidjson_SOURCE_DIR}/include
)

target_link_libraries(
	viewer
	PRIVATE
	ponder-flags
	vkof
	srat
	mor
	ponder
	stage
	Vulkan::Vulkan
	vk-bootstrap::vk-bootstrap
	glfw
	GPUOpen::VulkanMemoryAllocator
)

target_link_options(
	viewer
	PRIVATE
	-fuse-ld=mold
	${san-link}
)

install(
	TARGETS viewer
	RUNTIME DESTINATION bin
)

set_target_properties(viewer PROPERTIES INSTALL_RPATH "$ORIGIN")

# ngx loads the ray reconstruction library from the executable's directory
if(PONDER_DLSS)
	add_custom_command(
		TARGET viewer POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${dlss-rr-library}" "$<TARGET_FILE_DIR:viewer>"
		VERBATIM
	)
	install(FILES "${dlss-rr-library}" DESTINATION bin)
endif()

# ------------------------------------------------------------------------------
# -- shaderd (offline compute pipeline compile daemon)
# ------------------------------------------------------------------------------

find_package(slang CONFIG QUIET HINTS $ENV{HOME}/.local/slang)

if(slang_FOUND)
	FetchContent_Declare(
		ftxui
		GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
		GIT_TAG v7.0.3
		GIT_SHALLOW TRUE
	)
	set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
	set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(ftxui)

	add_executable(
		shaderd
		app/shaderd/main.cpp
		app/shaderd/compile.cpp
		app/shaderd/monitor.cpp
		app/shaderd/tui.cpp
		# vkof's screenshot needs tinyexr, implemented in ponder
		lib/ponder/src/tinyexr-impl.cpp
	)

	target_include_directories(
		shaderd
		PRIVATE
		${rapidjson_SOURCE_DIR}/include
	)

	target_include_directories(
		shaderd
		SYSTEM PRIVATE
		${stb_SOURCE_DIR}
		${tinyexr_SOURCE_DIR}
	)

	target_link_libraries(
		shaderd
		PRIVATE
		ponder-flags
		vkof
		mor
		srat
		slang::slang
		ftxui::screen
		ftxui::dom
		ftxui::component
	)

	target_link_options(
		shaderd
		PRIVATE
		-fuse-ld=mold
		${san-link}
	)

	install(
		TARGETS shaderd
		RUNTIME DESTINATION bin
	)

	set_target_properties(
		shaderd
		PROPERTIES
		INSTALL_RPATH_USE_LINK_PATH ON
	)
else()
	message(STATUS "slang sdk not found; shaderd is not built")
endif()

# ------------------------------------------------------------------------------
# -- tools
# ------------------------------------------------------------------------------

if(VKOF_AFTERMATH)
	add_executable(aftermath-decode lib/tools/aftermath-decode.cpp)

	target_link_libraries(
		aftermath-decode
		PRIVATE
		ponder-flags
		srat
		Vulkan::Vulkan
		Aftermath::Aftermath
	)

	target_link_options(
		aftermath-decode
		PRIVATE
		${san-link}
		-fuse-ld=mold
	)

	install(
		TARGETS aftermath-decode
		RUNTIME DESTINATION bin
	)

	set_target_properties(aftermath-decode PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

# ------------------------------------------------------------------------------
# -- unit tests
# ------------------------------------------------------------------------------

add_subdirectory(unit-test)
