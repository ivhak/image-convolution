#include "hip/hip_runtime.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>

extern "C" {
#include "../libs/bitmap.h"
}


#define BLOCK_X 32
#define BLOCK_Y 32

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5

float sobelYFilter[] = {-1, -2, -1,
                       0,  0,  0,
                       1,  2,  1};

float sobelXFilter[] = {-1, -0, 1,
                      -2,  0, 2,
                      -1,  0, 1};

float laplacian1Filter[] = { -1,  -4,  -1,
                           -4,  20,  -4,
                           -1,  -4,  -1};

float laplacian2Filter[] = { 0,  1,  0,
                           1, -4,  1,
                           0,  1,  0};

float laplacian3Filter[] = { -1,  -1,  -1,
                           -1,   8,  -1,
                           -1,  -1,  -1};

float gaussianFilter[] = { 1,  4,  6,  4, 1,
                         4, 16, 24, 16, 4,
                         6, 24, 36, 24, 6,
                         4, 16, 24, 16, 4,
                         1,  4,  6,  4, 1 };

const char* filterNames[]       = { "SobelY",     "SobelX",     "Laplacian 1",    "Laplacian 2",    "Laplacian 3",    "Gaussian"     };
float* const filters[]            = { sobelYFilter, sobelXFilter, laplacian1Filter, laplacian2Filter, laplacian3Filter, gaussianFilter };
unsigned int const filterDims[] = { 3,            3,            3,                3,                3,                5              };
float const filterFactors[]     = { 1.0,          1.0,          1.0,              1.0,              1.0,              1.0 / 256.0    };

int const maxFilterIndex = sizeof(filterDims) / sizeof(unsigned int);

void cleanup(char** input, char** output) {
    if (*input)
        free(*input);
    if (*output)
        free(*output);
}

void graceful_exit(char** input, char** output) {
    cleanup(input, output);
    exit(0);
}

void error_exit(char** input, char** output) {
    cleanup(input, output);
    exit(1);
}

// Helper function to swap bmpImageChannel pointers
void swapImageRawdata(pixel **one, pixel **two) {
    pixel *helper = *two;
    *two = *one;
    *one = helper;
}

void swapImage(bmpImage **one, bmpImage **two) {
    bmpImage *helper = *two;
    *two = *one;
    *one = helper;
}

__constant__ __device__ float d_filter[25];
// Apply convolutional filter on image data
__global__
void applyFilter(pixel *out, pixel *in, unsigned int width, unsigned int height, unsigned int filterDim, float filterFactor) {
    unsigned int global_x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int global_y = threadIdx.y + blockIdx.y*blockDim.y;

    if (global_x >= width || global_y >= height)  return;

    const int padding = (filterDim-1)/2;


    // All pixels needed for this block, including the halo.
    __shared__ pixel shared_in[BLOCK_X+4][BLOCK_Y+4];

    const bool P_W = threadIdx.x == 0             && blockIdx.x > 0;
    const bool P_N = threadIdx.y == 0             && blockIdx.y > 0;
    const bool P_E = threadIdx.x == blockDim.x -1 && blockIdx.x < gridDim.x -1;
    const bool P_S = threadIdx.y == blockDim.y -1 && blockIdx.y < gridDim.y -1;

    unsigned int x = threadIdx.x + padding;
    unsigned int y = threadIdx.y + padding;

    // Fill in the halo. Each thread fill in the pixel it's index points to. If
    // it is a border thread, it also has to load the `padding` amount of
    // pixels in its border direction. Corner threads have to fill in pixels
    // in both border directions, as well as the diagonal.
    //
    // Non-corner threads only have to copy `padding+1` number of pixels,
    // and the corner threads have to copy `padding+1`^2. With the filters used
    // in this program, this maxes out at 3 pixels for the non-corner threads,
    // and 9 for the corner threads.

    shared_in[y][x] = in[global_y*width + global_x];
    for (int i = 1; i < padding+1; i++) {
        if (P_W) shared_in[y][x-i] = in[global_y*width     + global_x - i];
        if (P_E) shared_in[y][x+i] = in[global_y*width     + global_x + i];
        if (P_N) shared_in[y-i][x] = in[(global_y-i)*width + global_x];
        if (P_S) shared_in[y+i][x] = in[(global_y+i)*width + global_x];

        /* north west */
        if (P_N && P_W){
            shared_in[y-i][x-i] = in[(global_y-i)*width + global_x - i];
            for (int j = 1; j < i; j++) {
                shared_in[y-i][x-i+j] = in[(global_y-i)*width + global_x - i+j];
                shared_in[y-i+j][x-i] = in[(global_y-i+j)*width + global_x - i];
            }
        }
        /* south west */
        if (P_S && P_W)  {
            shared_in[y+i][x-i] = in[(global_y+i)*width + global_x - i];
            for (int j = 1; j < i; j++) {
                shared_in[y+i][x-i+j] = in[(global_y+i)*width + global_x - i+j];
                shared_in[y+i-j][x-i] = in[(global_y+i-j)*width + global_x - i];
            }
        }
        /* north east */
        if (P_N && P_E) {
            shared_in[y-i][x+i] = in[(global_y-i)*width + global_x + i];
            for (int j = 1; j < i; j++) {
                shared_in[y-i][x+i-j] = in[(global_y-i)*width + global_x + i-j];
                shared_in[y-i+j][x+i] = in[(global_y-i+j)*width + global_x + i];
            }
        }
        /* south east*/
        if (P_S && P_E) {
            shared_in[y+i][x+i] = in[(global_y+i)*width + global_x + i];
            for (int j = 1; j < i; j++) {
                shared_in[y+i][x+i-j] = in[(global_y+i)*width + global_x + i-j];
                shared_in[y+i-j][x+i] = in[(global_y+i-j)*width + global_x + i];
            }
        }
    }
    __syncthreads();

    unsigned int const filterCenter = (filterDim / 2);
    int ar = 0, ag = 0, ab = 0;
    for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
            int nkx = filterDim - 1 - kx;

            int yy = y + (ky - filterCenter);
            int xx = x + (kx - filterCenter);
            int global_yy = global_y + (ky - filterCenter);
            int global_xx = global_x + (kx - filterCenter);
            if (global_xx >= 0 && global_xx < (int) width && global_yy >=0 && global_yy < (int) height) {
                ar += shared_in[yy][xx].r * d_filter[nky * filterDim + nkx];
                ag += shared_in[yy][xx].g * d_filter[nky * filterDim + nkx];
                ab += shared_in[yy][xx].b * d_filter[nky * filterDim + nkx];
            }
        }
    }

    ar *= filterFactor;
    ag *= filterFactor;
    ab *= filterFactor;
    ar = (ar < 0) ? 0 : ar;
    ag = (ag < 0) ? 0 : ag;
    ab = (ab < 0) ? 0 : ab;

    out[global_y*width + global_x].r = (ar > 255) ? 255 : ar;
    out[global_y*width + global_x].g = (ag > 255) ? 255 : ag;
    out[global_y*width + global_x].b = (ab > 255) ? 255 : ab;
}

void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -k, --filter     <filter>        filter index (0<=x<=%u) (2)\n", maxFilterIndex -1);
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s before.bmp after.bmp -i 10000\n", exec);
}

int main(int argc, char **argv) {
    /*
       Parameter parsing, don't change this!
     */
    unsigned int iterations = 1;
    char *output = NULL;
    char *input = NULL;
    unsigned int filterIndex = 2;

    static struct option const long_options[] =  {
        {"help",       no_argument,       0, 'h'},
        {"filter",     required_argument, 0, 'k'},
        {"iterations", required_argument, 0, 'i'},
        {0, 0, 0, 0}
    };

    static char const * short_options = "hk:i:";
    {
        char *endptr;
        int c;
        int parse;
        int option_index = 0;
        while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
            switch (c) {
                case 'h':
                    help(argv[0],0, NULL);
                    graceful_exit(&input,&output);
                case 'k':
                    parse = strtol(optarg, &endptr, 10);
                    if (endptr == optarg || parse < 0 || parse >= maxFilterIndex) {
                        help(argv[0], c, optarg);
                        error_exit(&input,&output);
                    }
                    filterIndex = (unsigned int) parse;
                    break;
                case 'i':
                    iterations = strtol(optarg, &endptr, 10);
                    if (endptr == optarg) {
                        help(argv[0], c, optarg);
                        error_exit(&input,&output);
                    }
                    break;
                default:
                    abort();
            }
        }
    }

    if (argc <= (optind+1)) {
        help(argv[0],' ',"Not enough arguments");
        error_exit(&input,&output);
    }

    unsigned int arglen = strlen(argv[optind]);
    input = (char*)calloc(arglen + 1, sizeof(char));
    strncpy(input, argv[optind], arglen);
    optind++;

    arglen = strlen(argv[optind]);
    output = (char*)calloc(arglen + 1, sizeof(char));
    strncpy(output, argv[optind], arglen);
    optind++;

    /*
       End of Parameter parsing!
     */


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

    printf("Apply filter '%s' on image with %u x %u pixels for %u iterations\n", filterNames[filterIndex], image->width, image->height, iterations);

    // Here we do the actual computation!
    // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
    // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
    bmpImage *processImage = newBmpImage(image->width, image->height);

    const size_t size_of_all_pixels = (image->width)*(image->height)*sizeof(pixel);
    const size_t size_of_filter = filterDims[filterIndex]*filterDims[filterIndex]*sizeof(int);

    // Allocate and copy over to device-side arrays
    pixel *d_image_rawdata;
    pixel *d_process_image_rawdata;

    // Allocate space for both device copies of the image, as well as for the filter.
    hipMalloc((void **)&d_process_image_rawdata, size_of_all_pixels);
    hipMalloc((void **)&d_image_rawdata,         size_of_all_pixels);

    // Set the device side arrays.
    hipMemset(d_process_image_rawdata, 0, size_of_all_pixels);
    hipMemcpy(d_image_rawdata, image->rawdata, size_of_all_pixels, hipMemcpyHostToDevice);
    hipMemcpyToSymbol(HIP_SYMBOL(d_filter), filters[filterIndex], size_of_filter);

    // We want one thread per pixel. Set the block size, and based on that set
    // the grid size to the smallest multiple of block size such that
    // block_size*grid_size >= width*height
    dim3 block_size(BLOCK_X, BLOCK_Y);
    dim3 grid_size((image->width  + block_size.x - 1) / block_size.x,
                   (image->height + block_size.y - 1) / block_size.y);

#if 0
    printf("Block size: %dx%d\n", block_size.x, block_size.y);
    printf("Grid size:  %dx%d\n", grid_size.x, grid_size.y);
    printf("# threads:  %d\n", block_size.x*grid_size.x*block_size.y*grid_size.y);
    printf("# pixels:   %d\n", image->width*image->height);
#endif

    // Start time measurement
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (unsigned int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(applyFilter, dim3(grid_size), dim3(block_size), 0, 0, d_process_image_rawdata,
                                               d_image_rawdata,
                                               image->width, image->height,
                                               filterDims[filterIndex],
                                               filterFactors[filterIndex]);
        // Swap the image and process_image
        swapImageRawdata(&d_image_rawdata, &d_process_image_rawdata);
    }
    // Copy back from the device-side array
    hipMemcpy(image->rawdata, d_image_rawdata, size_of_all_pixels, hipMemcpyDeviceToHost);

    // Stop the timer; calculate and print the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float spentTime = ((end_time.tv_sec - start_time.tv_sec)) + ((end_time.tv_nsec - start_time.tv_nsec)) * 1e-9;
    printf("Time spent: %.3f seconds\n", spentTime);
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

    printf("Launched blocks of size %d. Theoretical occupancy: %f\n", block_size.x*block_size.y, occupancy);

    hipFree(d_image_rawdata);
    hipFree(d_process_image_rawdata);
    freeBmpImage(processImage);
    //Write the image back to disk
    if (saveBmpImage(image, output) != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", output);
        freeBmpImage(image);
        error_exit(&input,&output);
    };

    graceful_exit(&input,&output);
};
