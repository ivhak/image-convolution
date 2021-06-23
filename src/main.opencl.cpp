#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include "CL/opencl.h"

extern "C" {
#include "../libs/bitmap.h"
#include "../libs/shared.h"
}


#define BLOCK_X 32
#define BLOCK_Y 32
#define MAX_SOURCE_SIZE (0x100000)

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

    printf("Apply filter '%s' on image with %u x %u pixels for %u iterations\n", filterNames[filterIndex], image->width, image->height, iterations);

    // Here we do the actual computation!
    // image->data is a 2-dimensional array of pixel which is accessed row first ([y][x])
    // each pixel is a struct of 3 unsigned char for the red, blue and green colour channel
    bmpImage *processImage = newBmpImage(image->width, image->height);

    const size_t size_of_all_pixels = (image->width)*(image->height)*sizeof(pixel);
    const size_t size_of_filter = filterDims[filterIndex]*filterDims[filterIndex]*sizeof(int);

    //OpenCL source can be placed in the source code as text strings or read from another file.
    FILE *fp;
    const char fileName[] = "./kernel.cl";
    size_t source_size;
    char *source_str;
 
    // read the kernel file into ram
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp );
    fclose( fp );

    // Find the available GPU
    const cl_uint num = 1;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 0, NULL, (cl_uint*)&num);

    cl_device_id devices[1];
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, num, devices, NULL);

    // Create a compute context with the GPU
    cl_context context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);

    // create a command queue
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_DEFAULT, 1, devices, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, devices[0], NULL, NULL);

    // allocate the buffer memory objects aka device side buffers
    cl_mem memobjs[] = { clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_of_all_pixels, NULL, NULL),
                         clCreateBuffer(context, CL_MEM_READ_WRITE,                        size_of_all_pixels, NULL, NULL)};

    // create the compute program
    // const char* fft1D_1024_kernel_src[1] = {  };
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)& source_str, NULL, NULL);

    // build the compute program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // create the compute kernel
    cl_kernel kernel = clCreateKernel(program, "kernel", NULL);

    // set the args values

    size_t local_work_size[1] = { 256 };

    clSetKernelArg(kernel, 0, sizeof(cl_mem),       (void *)&memobjs[0]);
    clSetKernelArg(kernel, 1, sizeof(cl_mem),       (void *)&memobjs[1]);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), (void *)&image->width);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *)&image->height);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), (void *)&filterDims[filterIndex]);
    clSetKernelArg(kernel, 5, sizeof(float),        (void *)&filterDims[filterIndex]);


    // create N-D range object with work-item dimensions and execute kernel
    size_t global_work_size[1] = { 256 };

    global_work_size[0] = image->width*image->height;
    local_work_size[0] = 64; //Nvidia: 192 or 256

    // Start time measurement
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    int swap = 1;
    for (unsigned int i = 0; i < iterations; i++) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        // Swap
        clSetKernelArg(kernel, swap, sizeof(cl_mem),       (void *)&memobjs[0]);
        clSetKernelArg(kernel, !swap, sizeof(cl_mem),      (void *)&memobjs[1]);
        swap = !swap;
    }
    // TODO: Copy back from the device-side array

    // Stop the timer; calculate and print the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float spentTime = ((end_time.tv_sec - start_time.tv_sec)) + ((end_time.tv_nsec - start_time.tv_nsec)) * 1e-9;
    printf("Time spent: %.3f seconds\n", spentTime);

    // TODO: Free

    freeBmpImage(processImage);
    //Write the image back to disk
    if (saveBmpImage(image, output) != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", output);
        freeBmpImage(image);
        error_exit(&input,&output);
    };

    graceful_exit(&input,&output);
};
