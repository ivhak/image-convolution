#include "hip/hip_runtime.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
#include "../../lib/bitmap.h"
#include "../../lib/shared.h"
}


#define BLOCK_X 32
#define BLOCK_Y 32

#define HIP_CHECK(command) {     \
    hipError_t status = command; \
    if (status!=hipSuccess) {     \
        printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__, hipGetErrorString(status)); \
        exit(1); \
    } \
}


__constant__ __device__ int d_filter[25];
// Apply convolutional filter on image data
__global__
void applyFilter(
        unsigned char *red_in, unsigned char *red_out,
        unsigned char *green_in, unsigned char *green_out,
        unsigned char *blue_in, unsigned char *blue_out,
        unsigned int width, unsigned int height, unsigned int filterDim, float filterFactor
) {
    unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x >= width || y >= height)  return;

    unsigned int const filterCenter = (filterDim / 2);
    int ar = 0, ag = 0, ab = 0;
    for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
            int nkx = filterDim - 1 - kx;

            int yy = y + (ky - filterCenter);
            int xx = x + (kx - filterCenter);
            if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
                ar += red_in  [yy*width+xx] * d_filter[nky * filterDim + nkx];
                ag += green_in[yy*width+xx] * d_filter[nky * filterDim + nkx];
                ab += blue_in [yy*width+xx] * d_filter[nky * filterDim + nkx];
            }
        }
    }

    ar *= filterFactor;
    ag *= filterFactor;
    ab *= filterFactor;
    ar = (ar < 0) ? 0 : ar;
    ag = (ag < 0) ? 0 : ag;
    ab = (ab < 0) ? 0 : ab;

    red_out  [y*width + x] = (ar > 255) ? 255 : ar;
    green_out[y*width + x] = (ag > 255) ? 255 : ag;
    blue_out [y*width + x] = (ab > 255) ? 255 : ab;
}

void swap_channel(unsigned char **in, unsigned char **out)
{
    unsigned char *tmp = *in;
    *in = *out;
    *out = tmp;
}

int main(int argc, char **argv) {
    /*
       Parameter parsing, don't change this!
     */
    unsigned int iterations = 1;
    char *output = NULL;
    char *input = NULL;
    unsigned int filterIndex = 2;

    parse_args(argc, argv, &iterations, &filterIndex, &output, &input);

    /*
       Create the BMP image and load it from disk.
     */
    bmpImage *image = newBmpImage(0,0);
    if (image == NULL) {
        fprintf(stderr, "Could not allocate new image!\n");
        error_exit(&input,&output);
    }

    if (loadBmpImage(image, input) != 0) {
        fprintf(stderr, "Could not load bmp image '%s'!\n", input);
        freeBmpImage(image);
        error_exit(&input,&output);
    }


    // Here we do the actual computation!
    // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
    // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
    bmpImage *processImage = newBmpImage(image->width, image->height);

    const size_t size_of_channel = (image->width)*(image->height)*sizeof(unsigned char);
    const size_t size_of_filter = filterDims[filterIndex]*filterDims[filterIndex]*sizeof(int);

    // Allocate and copy over to device-side arrays
    // pixel *d_image_rawdata;
    // pixel *d_process_image_rawdata;

    bmpImageChannel *image_channel_r   = newBmpImageChannel(image->width, image->height);
    bmpImageChannel *image_channel_g = newBmpImageChannel(image->width, image->height);
    bmpImageChannel *image_channel_b  = newBmpImageChannel(image->width, image->height);

    extractImageChannel(image_channel_r,   image, extractRed);
    extractImageChannel(image_channel_g, image, extractGreen);
    extractImageChannel(image_channel_b,  image, extractBlue);

    unsigned char *d_channel_in_r;
    unsigned char *d_channel_out_r;
    unsigned char *d_channel_in_b;
    unsigned char *d_channel_out_b;
    unsigned char *d_channel_in_g;
    unsigned char *d_channel_out_g;

    // Allocate space for both device copies of the image, as well as for the filter.
    HIP_CHECK(hipMalloc((void **)&d_channel_in_r,  size_of_channel));
    HIP_CHECK(hipMalloc((void **)&d_channel_out_r, size_of_channel));

    HIP_CHECK(hipMalloc((void **)&d_channel_in_b,  size_of_channel));
    HIP_CHECK(hipMalloc((void **)&d_channel_out_b, size_of_channel));

    HIP_CHECK(hipMalloc((void **)&d_channel_in_g,  size_of_channel));
    HIP_CHECK(hipMalloc((void **)&d_channel_out_g, size_of_channel));

    // Set the device side arrays.
    HIP_CHECK(hipMemcpy(d_channel_in_r, image_channel_r->rawdata, size_of_channel, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_channel_in_g, image_channel_g->rawdata, size_of_channel, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_channel_in_b, image_channel_b->rawdata, size_of_channel, hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_filter), filters[filterIndex], size_of_filter));

    // We want one thread per pixel. Set the block size, and based on that set
    // the grid size to the smallest multiple of block size such that
    // block_size*grid_size >= width*height
    dim3 block_size(BLOCK_X, BLOCK_Y);
    dim3 grid_size((image->width  + block_size.x - 1) / block_size.x,
                   (image->height + block_size.y - 1) / block_size.y);

    // Start time measurement
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (unsigned int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(applyFilter, dim3(grid_size), dim3(block_size), 0, 0,
                           d_channel_in_r,   d_channel_out_r,
                           d_channel_in_g, d_channel_out_g,
                           d_channel_in_b,  d_channel_out_b,
                           image->width, image->height,
                           filterDims[filterIndex], filterFactors[filterIndex]);
        HIP_CHECK(hipGetLastError());
        // Swap the image and process_image
        swap_channel(&d_channel_in_r,  &d_channel_out_r);
        swap_channel(&d_channel_in_g, &d_channel_out_g);
        swap_channel(&d_channel_in_b, &d_channel_out_b);
    }
    // Copy back from the device-side array
    // HIP_CHECK(hipMemcpy(image->rawdata, d_image_rawdata, size_of_all_pixels, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(image_channel_r->rawdata, d_channel_in_r, size_of_channel, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(image_channel_g->rawdata, d_channel_in_g, size_of_channel, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(image_channel_b->rawdata, d_channel_in_b, size_of_channel, hipMemcpyDeviceToHost));

    // Stop the timer; calculate and print the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float spentTime = ((end_time.tv_sec - start_time.tv_sec)) + ((end_time.tv_nsec - start_time.tv_nsec)) * 1e-9;

    log_execution(filterNames[filterIndex], image->width, image->height, iterations, spentTime);

#ifdef VERBOSE
    // calculate theoretical occupancy
    int max_active_blocks;
    int block_size_1d = block_size.x*block_size.y;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                  applyFilter,
                                                  block_size_1d,
                                                  0);

    int device;
    hipDeviceProp_t props;
    hipGetDevice(&device);
    hipGetDeviceProperties(&props, device);
    float occupancy = (max_active_blocks * block_size_1d / props.warpSize) /
                      (float)(props.maxThreadsPerMultiProcessor / props.warpSize);

    printf("Max active blocks: %d\n", max_active_blocks);
    printf("Max threads per multiprocessor: %d\n", props.maxThreadsPerMultiProcessor);
    printf("Launched blocks of size %d. Theoretical occupancy: %f\n", block_size.x*block_size.y, occupancy);
#endif

    hipFree(d_channel_in_r);
    hipFree(d_channel_out_r);
    hipFree(d_channel_in_g);
    hipFree(d_channel_out_g);
    hipFree(d_channel_in_b);
    hipFree(d_channel_out_b);

    for (int i = 0; i < image->height; i++)
        for (int j = 0; j < image->width; j++) {
            image->data[i][j].b = image_channel_b->data[i][j];
            image->data[i][j].g = image_channel_g->data[i][j];
            image->data[i][j].r = image_channel_r->data[i][j];
        }

    //Write the image back to disk
    if (saveBmpImage(image, output) != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", output);
        freeBmpImage(image);
        error_exit(&input,&output);
    };

    graceful_exit(&input,&output);
};
