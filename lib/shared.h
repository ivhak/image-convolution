#ifndef SHARED_H
#define SHARED_H
#include "bitmap.h"
#include <time.h>

extern const int sobelYFilter[];
extern const int sobelXFilter[];
extern const int laplacian1Filter[];
extern const int laplacian2Filter[];
extern const int laplacian3Filter[];
extern const int gaussianFilter[];
extern const char* filterNames[];
extern const int* filters[];
extern const unsigned int filterDims[] ;
extern const float filterFactors[];
extern const int maxFilterIndex;

void cleanup(char** input, char** output);
void graceful_exit(char** input, char** output);
void error_exit(char** input, char** output);

void help(char const *exec, char const opt, char const *optarg);
void parse_args(int argc, char **argv, unsigned *iterations, unsigned *filterIndex, char**output, char **input);

// Helper function to swap bmpImageChannel pointers
void swapImageRawdata(pixel **one, pixel **two);
void swapImage(bmpImage **one, bmpImage **two);

float time_spent(struct timespec t0, struct timespec t1);

void log_execution(const char *filter_name, unsigned width, unsigned height, unsigned iterations, float spent_time);

#endif
