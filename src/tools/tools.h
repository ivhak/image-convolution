#ifndef TOOLS_H
#define TOOLS_H

#include <stdio.h>

#include "../../lib/bitmap.h"

image_t *image_from_filename(const char *filename)
{
    image_t *image = new_image(0,0);
    if (image == NULL) {
        fprintf(stderr, "Could not allocate new image!\n");
    }

    if (load_image(image, filename) != 0) {
        fprintf(stderr, "Could not load bmp image '%s'!\n", filename);
        free_image(image);
    }
    return image;
}
#endif
