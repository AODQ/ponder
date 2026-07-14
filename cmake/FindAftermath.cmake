# FindAftermath.cmake
# Finds the NVIDIA Aftermath GPU crash dump SDK.
#
# Searches in:
#   ${CMAKE_SOURCE_DIR}/third-party/aftermath
#   $ENV{AFTERMATH_SDK_DIR}
#
# On success sets:
#   Aftermath_FOUND
#   Aftermath_INCLUDE_DIR
#   Aftermath_LIBRARY
# and creates imported target: Aftermath::Aftermath

set(_aftermath_hints
	"${CMAKE_SOURCE_DIR}/third-party/aftermath"
	"$ENV{AFTERMATH_SDK_DIR}"
)

find_path(
	Aftermath_INCLUDE_DIR
	NAMES GFSDK_Aftermath.h
	HINTS ${_aftermath_hints}
	PATH_SUFFIXES include
)

find_library(
	Aftermath_LIBRARY
	NAMES GFSDK_Aftermath_Lib GFSDK_Aftermath_Lib.x64
	HINTS ${_aftermath_hints}
	PATH_SUFFIXES lib lib/x64 lib/linux-x64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
	Aftermath
	REQUIRED_VARS Aftermath_INCLUDE_DIR Aftermath_LIBRARY
)

if(Aftermath_FOUND AND NOT TARGET Aftermath::Aftermath)
	add_library(Aftermath::Aftermath UNKNOWN IMPORTED)
	set_target_properties(
		Aftermath::Aftermath
		PROPERTIES
		IMPORTED_LOCATION "${Aftermath_LIBRARY}"
		INTERFACE_INCLUDE_DIRECTORIES "${Aftermath_INCLUDE_DIR}"
	)
endif()
