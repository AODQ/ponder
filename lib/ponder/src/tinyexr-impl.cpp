#define TINYEXR_IMPLEMENTATION
// use stb_image's zlib instead of the bundled miniz; stb is already a
// dependency and tinyexr's miniz copy lives outside its include dir
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
// this project builds with CMAKE_COMPILE_WARNING_AS_ERROR on, but the
// vendored tinyexr/stb_image headers aren't written to satisfy it; a newer
// gcc's -Wstringop-overflow false-positives inside libstdc++'s std::fill,
// called from tinyexr's implementation, otherwise hard-fails the build
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#include <stb_image.h>
#include <tinyexr.h>
#pragma GCC diagnostic pop
