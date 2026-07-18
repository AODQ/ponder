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
	GIT_TAG        v1.14
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

target_include_directories(
	vkof
	SYSTEM PUBLIC
	${imgui_SOURCE_DIR}
	PRIVATE
	${vma_SOURCE_DIR}/include
	${stb_SOURCE_DIR}
	${tinyexr_SOURCE_DIR}
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
)

target_include_directories(
	ponder
	PUBLIC
	lib/ponder/include
)

target_include_directories(
	ponder
	SYSTEM PRIVATE
	${stb_SOURCE_DIR}
	${tinyexr_SOURCE_DIR}
)

target_link_libraries(
	ponder
	PUBLIC
	vkof
	PRIVATE
	ponder-flags
	openvdb-system
)

# nanovdb is header-only; this unlocks its openvdb-conversion code paths
target_compile_definitions(ponder PRIVATE NANOVDB_USE_OPENVDB)

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
# -- viewer (opens gltf files and renders them with the ponder path tracer)
# ------------------------------------------------------------------------------

add_executable(viewer app/viewer/main.cpp)

target_compile_definitions(
	viewer
	PRIVATE
	PONDER_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/shaders/"
	PONDER_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/ponder/assets/"
	MOR_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/lib/mor/include/"
)

target_include_directories(
	viewer
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/app/viewer
)

target_include_directories(
	viewer
	SYSTEM PRIVATE
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
