#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "shared.h"
#include "bitmap.h"


// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5

const int sobelYFilter[] = {-1, -2, -1,
                       0,  0,  0,
                       1,  2,  1};

const int sobelXFilter[] = {-1, -0, 1,
                      -2,  0, 2,
                      -1,  0, 1};

const int laplacian1Filter[] = { -1,  -4,  -1,
                           -4,  20,  -4,
                           -1,  -4,  -1};

const int laplacian2Filter[] = { 0,  1,  0,
                           1, -4,  1,
                           0,  1,  0};

const int laplacian3Filter[] = { -1,  -1,  -1,
                           -1,   8,  -1,
                           -1,  -1,  -1};

const int gaussianFilter[] = { 1,  4,  6,  4, 1,
                         4, 16, 24, 16, 4,
                         6, 24, 36, 24, 6,
                         4, 16, 24, 16, 4,
                         1,  4,  6,  4, 1 };

const char* filterNames[]       = { "SobelY",     "SobelX",     "Laplacian 1",    "Laplacian 2",    "Laplacian 3",    "Gaussian"     };
const int* filters[]            = { sobelYFilter, sobelXFilter, laplacian1Filter, laplacian2Filter, laplacian3Filter, gaussianFilter };
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

void parse_args(int argc, char **argv, unsigned *iterations, unsigned *filterIndex, char**output, char **input)
{
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
                    graceful_exit(input,output);
                case 'k':
                    parse = strtol(optarg, &endptr, 10);
                    if (endptr == optarg || parse < 0 || parse >= maxFilterIndex) {
                        help(argv[0], c, optarg);
                        error_exit(input,output);
                    }
                    *filterIndex = (unsigned int) parse;
                    break;
                case 'i':
                    *iterations = strtol(optarg, &endptr, 10);
                    if (endptr == optarg) {
                        help(argv[0], c, optarg);
                        error_exit(input,output);
                    }
                    break;
                default:
                    abort();
            }
        }
    }

    if (argc <= (optind+1)) {
        help(argv[0],' ',"Not enough arguments");
        error_exit(input,output);
    }

    unsigned int arglen = strlen(argv[optind]);
    *input = (char*)calloc(arglen + 1, sizeof(char));
    strncpy(*input, argv[optind], arglen);
    optind++;

    arglen = strlen(argv[optind]);
    *output = (char*)calloc(arglen + 1, sizeof(char));
    strncpy(*output, argv[optind], arglen);
    optind++;

}

float time_spent(struct timespec t0, struct timespec t1)
{
    return ((t1.tv_sec - t0.tv_sec)) + ((t1.tv_nsec - t0.tv_nsec)) * 1e-9;
}

void log_execution(const char *filter_name, unsigned width, unsigned height, unsigned iterations, float spent_time) {

    printf("%-11s, %u x %u pixels, %4u iterations: %.6f seconds\n", filter_name, width, height, iterations, spent_time);
}
