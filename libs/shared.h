#ifndef SHARED_H
#define SHARED_H
#include "bitmap.h"
extern const float sobelYFilter[];
extern const float sobelXFilter[];
extern const float laplacian1Filter[];
extern const float laplacian2Filter[];
extern const float laplacian3Filter[];
extern const float gaussianFilter[];
extern const char* filterNames[];
extern const float* filters[];
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

#endif
