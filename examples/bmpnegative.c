#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>
#include "../lib/bitmap.h"
#include "../src/tools/tools.h"

int main(int argc, char *argv[])
{
    char *filename_in  = argv[1];
    char *filename_out = argv[2];

    image_t *in = image_from_filename(filename_in);
    image_t *out = new_image(in->width, in->height);

    const int width  = in->width;
    const int height = in->height;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            out->rawdata[i*width+j].r = 255 - in->rawdata[i*width+j].r;
            out->rawdata[i*width+j].g = 255 - in->rawdata[i*width+j].g;
            out->rawdata[i*width+j].b = 255 - in->rawdata[i*width+j].b;

        }
    }
    save_image(out, filename_out);
}
