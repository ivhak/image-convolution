#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>
#include "../../lib/bitmap.h"
#include "tools.h"

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: bmptile <X> <Y> <input file>.bmp <output file>.bmp\n");
        exit(1);
    }
    int tiles_x_direction = atoi(argv[1]);
    int tiles_y_direction = atoi(argv[2]);
    char *filename_in  = argv[3];
    char *filename_out = argv[4];

    image_t *original = image_from_filename(filename_in);

    image_t *new = new_image(tiles_x_direction*original->width, tiles_y_direction*original->height);

    for (int i = 0; i < new->height; i++)
        for (int j = 0; j < new->width; j++) {
            new->data[i][j] = original->data[i%(original->height)][j%(original->width)];
        }
    save_image(new, filename_out);

}
