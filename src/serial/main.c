#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

#include "../../lib/bitmap.h"
#include "../../lib/shared.h"
#include "../../lib/filter.h"


// Apply convolutional kernel on image data
void applyKernel(pixel **out, pixel **in, unsigned int width, unsigned int height, const int *kernel, unsigned int kernelDim, float kernelFactor) {
    unsigned int const kernelCenter = (kernelDim / 2);
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int ar = 0, ag = 0, ab = 0;
            for (unsigned int ky = 0; ky < kernelDim; ky++) {
                int nky = kernelDim - 1 - ky;
                for (unsigned int kx = 0; kx < kernelDim; kx++) {
                    int nkx = kernelDim - 1 - kx;

                    int yy = y + (ky - kernelCenter);
                    int xx = x + (kx - kernelCenter);
                    if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
                        ar += in[yy][xx].r * kernel[nky * kernelDim + nkx];
                        ag += in[yy][xx].g * kernel[nky * kernelDim + nkx];
                        ab += in[yy][xx].b * kernel[nky * kernelDim + nkx];
                    }
                }
            }
            if (ar || ag || ab) {
                ar *= kernelFactor;
                ag *= kernelFactor;
                ab *= kernelFactor;
                out[y][x].r = (ar > 255) ? 255 : ar;
                out[y][x].g = (ag > 255) ? 255 : ag;
                out[y][x].b = (ab > 255) ? 255 : ab;
            } else {
                out[y][x].r = 0;
                out[y][x].g = 0;
                out[y][x].b = 0;
            }
        }
    }
}

int main(int argc, char **argv) {
    unsigned int iterations = 1;
    char *output = NULL;
    char *input = NULL;
    unsigned int filter_index = 2;
    int ret = 0;

    parse_args(argc, argv, &iterations, &filter_index, &output, &input);

    // Create the BMP image and load it from disk.
    image_t *image = new_image(0,0);
    if (image == NULL) {
        fprintf(stderr, "Could not allocate new image!\n");
        goto error_exit;
    }

    if (load_image(image, input) != 0) {
        fprintf(stderr, "Could not load bmp image '%s'!\n", input);
        free_image(image);
        goto error_exit;
    }



    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Here we do the actual computation!
    // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
    // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
    image_t *processImage = new_image(image->width, image->height);
    for (unsigned int i = 0; i < iterations; i ++) {
        applyKernel(processImage->data,
                image->data,
                image->width,
                image->height,
                filters[filter_index],
                filter_dimensions[filter_index],
                filter_factors[filter_index]
                );
        swap_image(&processImage, &image);
    }
    free_image(processImage);

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float execution_time = time_spent(start_time, end_time);
    log_execution(filter_names[filter_index], image->width, image->height, iterations, execution_time);


    //Write the image back to disk
    if (save_image(image, output) != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", output);
        free_image(image);
        goto error_exit;
    };

graceful_exit:
    ret = 0;
error_exit:
    if (input)
        free(input);
    if (output)
        free(output);
    return ret;
};
