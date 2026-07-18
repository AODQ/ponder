#define TINYEXR_IMPLEMENTATION
// use stb_image's zlib instead of the bundled miniz; stb is already a
// dependency and tinyexr's miniz copy lives outside its include dir
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#include <stb_image.h>
#include <tinyexr.h>
