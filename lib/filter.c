#include "filter.h"

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
