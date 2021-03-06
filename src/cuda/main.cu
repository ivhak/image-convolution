#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
#include "../../lib/bitmap.h"
#include "../../lib/shared.h"
#include "../../lib/filter.h"
}

#define BLOCK_X 32
#define BLOCK_Y 32

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
        if (abort) exit(code);

    }
}

typedef struct {
    unsigned char *r;
    unsigned char *g;
    unsigned char *b;
} d_image_channels;

// Store the filter to be used in device memory so that it does not have to be
// read in each iteration
__constant__ __device__ int d_filter[25];

// Apply convolutional filter on image data
__global__
void apply_filter(d_image_channels in, d_image_channels out,
                 unsigned int width, unsigned int height, unsigned int filterDim, float filterFactor)
{
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

int main(int argc, char **argv) {

    // Parse command line arguments; number of iterations, kernel to use, input file, output file
    unsigned int iterations = 1;
    char *output = NULL;
    char *input = NULL;
    unsigned int filter_index = 2;

    parse_args(argc, argv, &iterations, &filter_index, &output, &input);

    // Load the image from disk
    image_t *image = new_image(0,0);
    if (image == NULL) {
        fprintf(stderr, "Could not allocate new image!\n");
        error_exit(&input,&output);
    }

    if (load_image(image, input) != 0) {
        fprintf(stderr, "Could not load bmp image '%s'!\n", input);
        free_image(image);
        error_exit(&input,&output);
    }


    const size_t size_of_channel = (image->width)*(image->height)*sizeof(unsigned char);
    const size_t size_of_filter = filter_dimensions[filter_index]*filter_dimensions[filter_index]*sizeof(int);

    // Convert the image for AOS ([r g b r g b r g b ...r g b ]) to SOA ([r r r ... r g g g ... g b b b ... b])
    // so that data is better coalesced when being processed on the GPU
    imageSOA_t *image_soa = new_imageSOA(image->width, image->height);
    image_to_imageSOA(image, image_soa);

    d_image_channels d_in, d_out;

    // Allocate space for both device copies of the image, as well as for the filter.
    cudaMalloc((void **)&d_in.r,  size_of_channel);
    cudaMalloc((void **)&d_out.r, size_of_channel);

    cudaMalloc((void **)&d_in.b,  size_of_channel);
    cudaMalloc((void **)&d_out.b, size_of_channel);

    cudaMalloc((void **)&d_in.g,  size_of_channel);
    cudaMalloc((void **)&d_out.g, size_of_channel);

    // Set the device side arrays.
    cudaMemcpy(d_in.r, image_soa->r, size_of_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in.g, image_soa->g, size_of_channel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in.b, image_soa->b, size_of_channel, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_filter, filters[filter_index], size_of_filter);

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
        apply_filter<<<grid_size, block_size>>>(d_in, d_out,
                                               image->width, image->height,
                                               filter_dimensions[filter_index],
                                               filter_factors[filter_index]);
        // Swap the input/output buffers
        swap_image_channels(&d_in.r, &d_out.r);
        swap_image_channels(&d_in.g, &d_out.g);
        swap_image_channels(&d_in.b, &d_out.b);
    }
    cudaDeviceSynchronize();
    // Stop the timer; calculate and print the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float execution_time = time_spent(start_time, end_time);
    log_execution(filter_names[filter_index], image->width, image->height, iterations, execution_time);

    // Check for error
    cudaError_t error = cudaGetLastError();
    cudaErrorCheck(error);

    // Copy back from the device-side array
    //
    // NOTE: d_in is actually holding the output, as the buffers were swapped
    // after the last iteration:
    cudaMemcpy(image_soa->r, d_in.r, size_of_channel, cudaMemcpyDeviceToHost);
    cudaMemcpy(image_soa->g, d_in.g, size_of_channel, cudaMemcpyDeviceToHost);
    cudaMemcpy(image_soa->b, d_in.b, size_of_channel, cudaMemcpyDeviceToHost);


#ifdef VERBOSE
    // calculate theoretical occupancy
    int max_active_blocks;
    int block_size_1d = block_size.x*block_size.y;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                  apply_filter,
                                                  block_size_1d,
                                                  0);

    int device;
    cudaDeviceProp_t props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    float occupancy = (max_active_blocks * block_size_1d / props.warpSize) /
                      (float)(props.maxThreadsPerMultiProcessor / props.warpSize);

    printf("Max active blocks: %d\n", max_active_blocks);
    printf("Max threads per multiprocessor: %d\n", props.maxThreadsPerMultiProcessor);
    printf("Launched blocks of size %d. Theoretical occupancy: %f\n", block_size.x*block_size.y, occupancy);
#endif

    cudaFree(d_in.r);
    cudaFree(d_out.r);
    cudaFree(d_in.g);
    cudaFree(d_out.g);
    cudaFree(d_in.b);
    cudaFree(d_out.b);

    // Copy the image back to AOS form so that it is easily written to disk
    imageSOA_to_image(image_soa, image);
    free_imageSOA(image_soa);

    // Write the image back to disk
    if (save_image(image, output) != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", output);
        free_image(image);
        error_exit(&input,&output);
    };

    graceful_exit(&input,&output);
};
