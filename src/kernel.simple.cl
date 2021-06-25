typedef struct pi {
    uchar b;
    uchar g;
    uchar r;
} pixel;

__kernel void applyFilter(__global pixel *out, __global pixel *in, __global int *filter,
                          uint width, uint height, uint filterDim, float filterFactor)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
#ifdef DEBUG
    if (x == 0 && y == 0) {

        printf("pixel[0].r: %d", in[0].r);
        printf("pixel[0].g: %d", in[0].g);
        printf("pixel[0].b: %d", in[0].b);
        printf("pixel[0].r: %d", in[1].r);
        printf("pixel[0].g: %d", in[1].g);
        printf("pixel[0].b: %d", in[1].b);
        printf("kernel: %d %d %d", filter[0], filter[1], filter[2]);
        printf("        %d %d %d", filter[3], filter[4], filter[5]);
        printf("        %d %d %d", filter[6], filter[7], filter[8]);
    }
#endif
    if (x >= width || y >= height) return;


    out[y*width + x].r = !in[y*width + x].r;
    out[y*width + x].g = !in[y*width + x].g;
    out[y*width + x].b = !in[y*width + x].b;
    return;


    uint filterCenter = (filterDim / 2);
    uint ar = 0, ag = 0, ab = 0;
    for (uint ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (uint kx = 0; kx < filterDim; kx++) {
            int nkx = filterDim - 1 - kx;

            int yy = y + (ky - filterCenter);
            int xx = x + (kx - filterCenter);
            if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
                ar += in[yy*width + xx].r * filter[nky * filterDim + nkx];
                ag += in[yy*width + xx].g * filter[nky * filterDim + nkx];
                ab += in[yy*width + xx].b * filter[nky * filterDim + nkx];
            }
        }
    }
    if (ar || ag || ab) {
        ar *= filterFactor;
        ag *= filterFactor;
        ab *= filterFactor;
        out[y*width +x].r = (ar > 255) ? 255 : ar;
        out[y*width +x].g = (ag > 255) ? 255 : ag;
        out[y*width +x].b = (ab > 255) ? 255 : ab;
    } else {
        out[y*width +x].r = 0;
        out[y*width +x].g = 0;
        out[y*width +x].b = 0;
    }
}
