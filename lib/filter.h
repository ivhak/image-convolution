#ifndef FILTER_H
#define FILTER_H
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
#endif
