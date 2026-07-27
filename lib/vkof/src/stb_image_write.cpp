// tinyexr's TINYEXR_USE_STB_ZLIB backend (lib/ponder/src/tinyexr-impl.cpp)
// borrows stb_image_write.h's deflate compressor instead of bundling
// miniz; this translation unit exists only to provide that symbol
// (stbi_zlib_compress), not for any of vkof's own image writing.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
