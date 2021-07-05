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


#define BLOCK_X 64
#define BLOCK_Y 16

#define HIP_CHECK(command) {     \
    hipError_t status = command; \
    if (status!=hipSuccess) {     \
        printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__, hipGetErrorString(status)); \
        exit(1); \
    } \
}

typedef struct {
    unsigned char *r;
    unsigned char *g;
    unsigned char *b;
} image_channels;


__constant__ __device__ int d_filter[25];
// Apply convolutional filter on image data
__global__
void applyFilter(image_channels in, image_channels out,
                 unsigned int width, unsigned int height, unsigned int filterDim, float filterFactor)
{
    unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x >= width || y >= height)  return;

    unsigned int const filterCenter = (filterDim / 2);
    int ar = 0, ag = 0, ab = 0;
    for (int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (int kx = 0; kx < filterDim; kx++) {
            int nkx = filterDim - 1 - kx;

            int yy = y + (ky - filterCenter);
            int xx = x + (kx - filterCenter);
            if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
                ar += in.r[yy*width+xx] * d_filter[nky * filterDim + nkx];
                ag += in.g[yy*width+xx] * d_filter[nky * filterDim + nkx];
                ab += in.b[yy*width+xx] * d_filter[nky * filterDim + nkx];
            }
        }
    }

    ar *= filterFactor;
    ag *= filterFactor;
    ab *= filterFactor;
    ar = (ar < 0) ? 0 : ar;
    ag = (ag < 0) ? 0 : ag;
    ab = (ab < 0) ? 0 : ab;

    out.r[y*width + x] = (ar > 255) ? 255 : ar;
    out.g[y*width + x] = (ag > 255) ? 255 : ag;
    out.b[y*width + x] = (ab > 255) ? 255 : ab;
}

void swap_image_channels(unsigned char **in, unsigned char **out)
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


    const size_t size_of_channel = (image->width)*(image->height)*sizeof(unsigned char);
    const size_t size_of_filter = filterDims[filterIndex]*filterDims[filterIndex]*sizeof(int);

    bmpImageChannel *image_channel_r = newBmpImageChannel(image->width, image->height);
    bmpImageChannel *image_channel_g = newBmpImageChannel(image->width, image->height);
    bmpImageChannel *image_channel_b = newBmpImageChannel(image->width, image->height);

    extractImageChannel(image_channel_r, image, extractRed);
    extractImageChannel(image_channel_g, image, extractGreen);
    extractImageChannel(image_channel_b, image, extractBlue);

    image_channels d_in, d_out;

    // Allocate space for both device copies of the image, as well as for the filter.
    HIP_CHECK(hipMalloc((void **)&d_in.r,  size_of_channel));
    HIP_CHECK(hipMalloc((void **)&d_out.r, size_of_channel));

    HIP_CHECK(hipMalloc((void **)&d_in.b,  size_of_channel));
    HIP_CHECK(hipMalloc((void **)&d_out.b, size_of_channel));

    HIP_CHECK(hipMalloc((void **)&d_in.g,  size_of_channel));
    HIP_CHECK(hipMalloc((void **)&d_out.g, size_of_channel));

    // Set the device side arrays.
    HIP_CHECK(hipMemcpy(d_in.r, image_channel_r->rawdata, size_of_channel, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_in.g, image_channel_g->rawdata, size_of_channel, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_in.b, image_channel_b->rawdata, size_of_channel, hipMemcpyHostToDevice));

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
                           d_in, d_out,
                           image->width, image->height,
                           filterDims[filterIndex], filterFactors[filterIndex]);
        HIP_CHECK(hipGetLastError());
        // Swap the image and process_image
        swap_image_channels(&d_in.r, &d_out.r);
        swap_image_channels(&d_in.g, &d_out.g);
        swap_image_channels(&d_in.b, &d_out.b);
    }
    hipDeviceSynchronize();
    // Stop the timer; calculate and print the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float execution_time = time_spent(start_time, end_time);
    log_execution(filterNames[filterIndex], image->width, image->height, iterations, execution_time);

    // Copy back from the device-side array
    // HIP_CHECK(hipMemcpy(image->rawdata, d_image_rawdata, size_of_all_pixels, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(image_channel_r->rawdata, d_in.r, size_of_channel, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(image_channel_g->rawdata, d_in.g, size_of_channel, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(image_channel_b->rawdata, d_in.b, size_of_channel, hipMemcpyDeviceToHost));



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

    hipFree(d_in.r);
    hipFree(d_in.g);
    hipFree(d_in.b);
    hipFree(d_out.r);
    hipFree(d_out.g);
    hipFree(d_out.b);

    for (int i = 0; i < image->height; i++)
        for (int j = 0; j < image->width; j++) {
            image->data[i][j].b = image_channel_b->data[i][j];
            image->data[i][j].g = image_channel_g->data[i][j];
            image->data[i][j].r = image_channel_r->data[i][j];
        }

    freeBmpImageChannel(image_channel_r);
    freeBmpImageChannel(image_channel_g);
    freeBmpImageChannel(image_channel_b);

    //Write the image back to disk
    if (saveBmpImage(image, output) != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", output);
        freeBmpImage(image);
        error_exit(&input,&output);
    };

    graceful_exit(&input,&output);
};
