#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

int vkof_write_png_uncompressed(
	char const * path, int w, int h, int comp,
	void const * data, int stride
) {
	stbi_write_png_compression_level = 0;
	return stbi_write_png(path, w, h, comp, data, stride);
}
