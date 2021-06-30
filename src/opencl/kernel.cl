#define BLOCK_X 16
#define BLOCK_Y 16

__kernel void applyFilter(
    __global unsigned char *in_r, __global unsigned char *out_r,
    __global unsigned char *in_g, __global unsigned char *out_g,
    __global unsigned char *in_b, __global unsigned char *out_b,
    __global int *filter,
    uint width, uint height, uint filterDim, float filterFactor)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    if (x >= width || y >= height)  return;

    const uint filterCenter = (filterDim / 2);
    int ar = 0, ag = 0, ab = 0;
    for (uint ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (uint kx = 0; kx < filterDim; kx++) {
            int nkx = filterDim - 1 - kx;
            int yy = y + (ky - filterCenter);
            int xx = x + (kx - filterCenter);
            if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height) {
                ar += in_r[yy*width+xx] * filter[nky * filterDim + nkx];
                ag += in_g[yy*width+xx] * filter[nky * filterDim + nkx];
                ab += in_b[yy*width+xx] * filter[nky * filterDim + nkx];
            }
        }
    }

    ar *= filterFactor;
    ag *= filterFactor;
    ab *= filterFactor;
    ar = (ar < 0) ? 0 : ar;
    ag = (ag < 0) ? 0 : ag;
    ab = (ab < 0) ? 0 : ab;

    out_r[y*width + x] = (ar > 255) ? 255 : ar;
    out_g[y*width + x] = (ag > 255) ? 255 : ag;
    out_b[y*width + x] = (ab > 255) ? 255 : ab;
}
