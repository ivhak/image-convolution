#ifndef FILTER_H
#define FILTER_H
extern const int sobelY_filter[];
extern const int sobelX_filter[];
extern const int laplacian1_filter[];
extern const int laplacian2_filter[];
extern const int laplacian3_filter[];
extern const int gaussian_filter[];

extern const char* filter_names[];
extern const int* filters[];

extern const unsigned int filter_dimensions[] ;
extern const float filter_factors[];
extern const int max_filter_index;
#endif
