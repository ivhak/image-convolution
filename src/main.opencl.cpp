#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include "CL/cl.h"

extern "C" {
#include "../libs/bitmap.h"
#include "../libs/shared.h"
}

#define BLOCK_X 16
#define BLOCK_Y 16
#define MAX_SOURCE_SIZE (0x100000)

#define DIE_IF_CL(err_code, str) do { if (err != CL_SUCCESS) {fprintf(stderr,"%d: %s\n", err_code, str); exit(1); } } while (0)

char *load_kernel_source(const char *filename);

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

    const size_t size_of_all_pixels = (image->width)*(image->height)*sizeof(pixel);
    const size_t size_of_filter = filterDims[filterIndex]*filterDims[filterIndex]*sizeof(int);

    // OpenCL source can be placed in the source code as text strings or read from another file.
    char *source_str = load_kernel_source("src/kernel.simple.cl");

    // Timing
    struct timespec start_time, end_time;

    // Setup OpenCL
    cl_platform_id    platform_id;
    cl_device_id      device_id;
    cl_context        context;
    cl_command_queue  queue;
    cl_program        program;
    cl_kernel         kernel1;
    cl_kernel         kernel2;

    cl_mem  d_image_data_in;
    cl_mem  d_image_data_out;
    cl_mem  d_filter_buf;

    cl_int err;


    err = clGetPlatformIDs(1, &platform_id, NULL);
    DIE_IF_CL(err, "Could not get platforms.");

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    DIE_IF_CL(err, "Could not get device.");

    // Create a compute context with the GPU
    context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
    DIE_IF_CL(err, "Could not create compute context.");

    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
    DIE_IF_CL(err, "Could not create command queue.");


    // Create the compute program
    program = clCreateProgramWithSource(context, 1, (const char **)& source_str, NULL, &err);
    DIE_IF_CL(err, "Could not create kernel program.");

    // Build the compute program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log;

        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
    }
    DIE_IF_CL(err, "Could not build kernel program.");


    // Allocate the buffer memory objects aka device side buffers
    // TODO: The data is not actually being copied over, only garbage data.
    d_image_data_in  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_of_all_pixels, (void*)image->rawdata, &err);
    DIE_IF_CL(err, "Could not create in buffer.");

    d_image_data_out = clCreateBuffer(context, CL_MEM_READ_WRITE, size_of_all_pixels, NULL, &err);
    DIE_IF_CL(err, "Could not create out buffer.");

    d_filter_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_of_filter, (void*)filters[filterIndex], &err);
    DIE_IF_CL(err, "Could not create kernel buffer.");

    // Create the compute kernels. Two mostly identical kernels are used, but
    // the image data in/out is swapped. This makes it easy to use the output
    // of the last iteration as the input of the next, without having to swap
    // buffers or reset kernel parameters.
    kernel1 = clCreateKernel(program, "applyFilter", &err);
    DIE_IF_CL(err, "Could not create kernel");

    kernel2 = clCreateKernel(program, "applyFilter", &err);
    DIE_IF_CL(err, "Could not create kernel");


    // Set the args
    err =  clSetKernelArg(kernel1, 0, sizeof(cl_mem),       &d_image_data_out); // output of kernel1
    err |= clSetKernelArg(kernel1, 1, sizeof(cl_mem),       &d_image_data_in);  // input  of kernel1
    err |= clSetKernelArg(kernel1, 2, sizeof(cl_mem),       &d_filter_buf);
    err |= clSetKernelArg(kernel1, 3, sizeof(unsigned int), &image->width);
    err |= clSetKernelArg(kernel1, 4, sizeof(unsigned int), &image->height);
    err |= clSetKernelArg(kernel1, 5, sizeof(unsigned int), &filterDims[filterIndex]);
    err |= clSetKernelArg(kernel1, 6, sizeof(float),        &filterFactors[filterIndex]);
    DIE_IF_CL(err, "Failed to set kernel argument.");

    err =  clSetKernelArg(kernel2, 0, sizeof(cl_mem),       &d_image_data_in);  // output of kernel2
    err |= clSetKernelArg(kernel2, 1, sizeof(cl_mem),       &d_image_data_out); // input  of kernel2
    err |= clSetKernelArg(kernel2, 2, sizeof(cl_mem),       &d_filter_buf);
    err |= clSetKernelArg(kernel2, 3, sizeof(unsigned int), &image->width);
    err |= clSetKernelArg(kernel2, 4, sizeof(unsigned int), &image->height);
    err |= clSetKernelArg(kernel2, 5, sizeof(unsigned int), &filterDims[filterIndex]);
    err |= clSetKernelArg(kernel2, 6, sizeof(float),        &filterFactors[filterIndex]);
    DIE_IF_CL(err, "Failed to set kernel argument.");

    // Create N-D range object with work-item dimensions and execute kernel
    // Round up the image width and height to the nearest multiple of BLOCK_X
    // and BLOCK_Y respectively.
    size_t global_work_size[2];
    global_work_size[0] = BLOCK_X*((image->width  + BLOCK_X - 1)/BLOCK_X);
    global_work_size[1] = BLOCK_Y*((image->height + BLOCK_Y - 1)/BLOCK_Y);

    size_t local_work_size[2] = {BLOCK_X, BLOCK_Y};


    // Start time measurement
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (unsigned int i = 0; i < iterations; i++) {
        err = clEnqueueNDRangeKernel(queue, i % 2 == 0 ? kernel1 : kernel2, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        DIE_IF_CL(err, "Failed to run kernel.");
        clFinish(queue);
    }

    clFinish(queue);

    // Copy back from the device-side array
    cl_mem d_actual_image_data_out = iterations % 2 == 1 ? d_image_data_in : d_image_data_out;
    err = clEnqueueReadBuffer(queue, d_actual_image_data_out, CL_TRUE, 0, size_of_all_pixels, image->rawdata, 0, NULL, NULL);
    DIE_IF_CL(err, "Failed to copy back to device.");

    clFinish(queue);


    // Stop the timer; calculate and print the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float spentTime = ((end_time.tv_sec - start_time.tv_sec)) + ((end_time.tv_nsec - start_time.tv_nsec)) * 1e-9;
    printf("Time spent: %.3f seconds\n", spentTime);

    //Write the image back to disk
    if (saveBmpImage(image, output) != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", output);
        freeBmpImage(image);
        error_exit(&input,&output);
    };

    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseMemObject(d_image_data_in);
    clReleaseMemObject(d_image_data_out);
    clReleaseMemObject(d_filter_buf);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    graceful_exit(&input,&output);
};

char *load_kernel_source(const char *filename)
{
    // OpenCL source can be placed in the source code as text strings or read from another file.
    FILE *fp;
    size_t source_size;
    char *source_str;

    // read the kernel file into ram
    fp = fopen(filename, "r");
    if (!fp) {
       perror("fopen");
       exit(1);
    }

    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp );
    fclose(fp);
    return source_str;
}
