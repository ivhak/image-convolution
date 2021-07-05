#include "filter.h"

// Convolutional _filter Examples, each with dimension 3,
// gaussian filter with dimension 5

const int sobelY_filter[]     = { -1, -2, -1,
                                   0,  0,  0,
                                   1,  2,  1 };

const int sobelX_filter[]     = { -1, -0,  1,
                                  -2,  0,  2,
                                  -1,  0,  1 };

const int laplacian1_filter[] = { -1, -4, -1,
                                  -4, 20, -4,
                                  -1, -4, -1 };

const int laplacian2_filter[] = {  0,  1,  0,
                                   1, -4,  1,
                                   0,  1,  0 };

const int laplacian3_filter[] = { -1, -1, -1,
                                  -1,  8, -1,
                                  -1, -1, -1 };

const int gaussian_filter[]   = {  1,  4,  6,  4, 1,
                                   4, 16, 24, 16, 4,
                                   6, 24, 36, 24, 6,
                                   4, 16, 24, 16, 4,
                                   1,  4,  6,  4, 1 };

const char         *filter_names[]      = { "SobelY",      "SobelX",      "Laplacian 1",     "Laplacian 2",     "Laplacian 3",     "Gaussian"      };
const int          *filters[]           = { sobelY_filter, sobelX_filter, laplacian1_filter, laplacian2_filter, laplacian3_filter, gaussian_filter };
const unsigned int  filter_dimensions[] = { 3,             3,             3,                 3,                 3,                 5               };
const float         filter_factors[]    = { 1.0,           1.0,           1.0,               1.0,               1.0,               1.0 / 256.0     };

int const max_filter_index = sizeof(filter_dimensions) / sizeof(unsigned int);
