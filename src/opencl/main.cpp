#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include "CL/cl.h"

extern "C" {
#include "../../lib/bitmap.h"
#include "../../lib/shared.h"
#include "../../lib/filter.h"
}

#define BLOCK_X 16
#define BLOCK_Y 16
#define MAX_SOURCE_SIZE (0x100000)

#define DIE_IF_CL(err_code, str) do { if (err != CL_SUCCESS) {fprintf(stderr,"%d: %s\n", err_code, str); exit(1); } } while (0)

typedef struct {
    cl_mem r;
    cl_mem g;
    cl_mem b;
} d_image_channels;

char *load_kernel_source(const char *filename);
void print_platform_info(cl_platform_id platform_id);
void print_device_info(cl_device_id device_id);


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
    bmp_image_t *image = new_bmp_image(0,0);
    if (image == NULL) {
        fprintf(stderr, "Could not allocate new image!\n");
        error_exit(&input,&output);
    }

    if (load_bmp_image(image, input) != 0) {
        fprintf(stderr, "Could not load bmp image '%s'!\n", input);
        free_bmp_image(image);
        error_exit(&input,&output);
    }


    const size_t size_of_channel = (image->width)*(image->height)*sizeof(unsigned char);
    const size_t size_of_filter = filterDims[filterIndex]*filterDims[filterIndex]*sizeof(int);

    bmp_image_soa_t *image_soa = new_bmp_image_soa(image->width, image->height);
    image_to_soa_image(image, image_soa);

    // OpenCL source can be placed in the source code as text strings or read from another file.
    char *source_str = load_kernel_source("src/opencl/kernel.cl");

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

    d_image_channels d_out;
    d_image_channels d_in;

    cl_mem  d_filter_buf;

    cl_int err;


    err = clGetPlatformIDs(1, &platform_id, NULL);
    DIE_IF_CL(err, "Could not get platforms.");

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    DIE_IF_CL(err, "Could not get device.");

    // print_platform_info(platform_id);
    // print_device_info(device_id);

    // Create a compute context with the GPU
    context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
    DIE_IF_CL(err, "Could not create compute context.");

    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
    DIE_IF_CL(err, "Could not create command queue.");


    // Create the compute program
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &err);
    DIE_IF_CL(err, "Could not create kernel program.");
    free(source_str);

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
    d_in.r  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_of_channel, (void*)image_soa->r, &err);
    d_in.g  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_of_channel, (void*)image_soa->g, &err);
    d_in.b  = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_of_channel, (void*)image_soa->b, &err);
    DIE_IF_CL(err, "Could not create in buffer.");

    d_out.r = clCreateBuffer(context, CL_MEM_READ_WRITE, size_of_channel, NULL, &err);
    d_out.g = clCreateBuffer(context, CL_MEM_READ_WRITE, size_of_channel, NULL, &err);
    d_out.b = clCreateBuffer(context, CL_MEM_READ_WRITE, size_of_channel, NULL, &err);
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
    err  = clSetKernelArg(kernel1, 0,   sizeof(cl_mem),         &d_in.r);
    err |= clSetKernelArg(kernel1, 1,   sizeof(cl_mem),         &d_out.r);
    err |= clSetKernelArg(kernel1, 2,   sizeof(cl_mem),         &d_in.g);
    err |= clSetKernelArg(kernel1, 3,   sizeof(cl_mem),         &d_out.g);
    err |= clSetKernelArg(kernel1, 4,   sizeof(cl_mem),         &d_in.b);
    err |= clSetKernelArg(kernel1, 5,   sizeof(cl_mem),         &d_out.b);
    err |= clSetKernelArg(kernel1, 6,   sizeof(cl_mem),         &d_filter_buf);
    err |= clSetKernelArg(kernel1, 7,   sizeof(unsigned int),   &image->width);
    err |= clSetKernelArg(kernel1, 8,   sizeof(unsigned int),   &image->height);
    err |= clSetKernelArg(kernel1, 9,   sizeof(unsigned int),   &filterDims[filterIndex]);
    err |= clSetKernelArg(kernel1, 10,  sizeof(float),          &filterFactors[filterIndex]);
    DIE_IF_CL(err, "Failed to set kernel argument.");

    err  = clSetKernelArg(kernel2, 0,   sizeof(cl_mem),         &d_out.r);
    err |= clSetKernelArg(kernel2, 1,   sizeof(cl_mem),         &d_in.r);
    err |= clSetKernelArg(kernel2, 2,   sizeof(cl_mem),         &d_out.g);
    err |= clSetKernelArg(kernel2, 3,   sizeof(cl_mem),         &d_in.g);
    err |= clSetKernelArg(kernel2, 4,   sizeof(cl_mem),         &d_out.b);
    err |= clSetKernelArg(kernel2, 5,   sizeof(cl_mem),         &d_in.b);
    err |= clSetKernelArg(kernel2, 6,   sizeof(cl_mem),         &d_filter_buf);
    err |= clSetKernelArg(kernel2, 7,   sizeof(unsigned int),   &image->width);
    err |= clSetKernelArg(kernel2, 8,   sizeof(unsigned int),   &image->height);
    err |= clSetKernelArg(kernel2, 9,   sizeof(unsigned int),   &filterDims[filterIndex]);
    err |= clSetKernelArg(kernel2, 10,  sizeof(float),          &filterFactors[filterIndex]);
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
    }
    clFinish(queue);
    // Stop the timer; calculate and print the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    float execution_time = time_spent(start_time, end_time);
    log_execution(filterNames[filterIndex], image->width, image->height, iterations, execution_time);

    // Copy back from the device-side array: Channel red
    d_image_channels *d_image_channels_out = iterations % 2 == 0 ? &d_in : &d_out;
    err  = clEnqueueReadBuffer(queue, d_image_channels_out->r, CL_TRUE, 0, size_of_channel, image_soa->r, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, d_image_channels_out->g, CL_TRUE, 0, size_of_channel, image_soa->g, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, d_image_channels_out->b, CL_TRUE, 0, size_of_channel, image_soa->b, 0, NULL, NULL);
    DIE_IF_CL(err, "Failed to copy back to device.");

    clFinish(queue);


    soa_image_to_image(image_soa, image);
    free_image_soa(image_soa);
    //Write the image back to disk
    if (save_bmp_image(image, output) != 0) {
        fprintf(stderr, "Could not save output to '%s'!\n", output);
        free_bmp_image(image);
        error_exit(&input,&output);
    };

    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseMemObject(d_in.r);
    clReleaseMemObject(d_out.r);
    clReleaseMemObject(d_in.g);
    clReleaseMemObject(d_out.g);
    clReleaseMemObject(d_in.b);
    clReleaseMemObject(d_out.b);
    clReleaseMemObject(d_filter_buf);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    graceful_exit(&input,&output);
};


void print_platform_info(cl_platform_id platform_id)
{
    char *profile = NULL;
    size_t size;
    clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, 0, profile, &size);
    profile = (char*)malloc(size);
    clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE,size, profile, NULL);
    printf("%s\n", profile);
    free(profile);
}

void print_device_info(cl_device_id device_id)
{
    size_t size;
    char *device = NULL;
    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 0, NULL, &size);
    device = (char*)malloc(sizeof(char)*size);
    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, size, device, NULL);
    printf("%s\n", device);
    free(device);
}

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
