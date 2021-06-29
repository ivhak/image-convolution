#define BLOCK_X 16
#define BLOCK_Y 16

typedef struct {
    uchar b;
    uchar g;
    uchar r;
} pixel;

// Multiplies A*x, leaving the result in y.
// A is a row-major matrix, meaning the (i,j) element is at A[i*ncols+j].
__kernel void applyFilter(__global pixel *out, __global pixel *in, __global int *filter,
                          uint width, uint height, uint filterDim, float filterFactor)
{
    size_t global_x = get_global_id(0);
    size_t global_y = get_global_id(1);
    if (global_x >= width || global_y >= height)  return;



#ifdef NO_SHARED_MEM
    __global pixel *shared_in = in;
    uint x = global_x;
    uint y = global_y;
#else
    // All pixels needed for this block, including the halo.
    __local pixel shared_in[BLOCK_X+4][BLOCK_Y+4];

    const int padding = (filterDim-1)/2;

    const size_t local_id_x   = get_local_id(0);
    const size_t local_id_y   = get_local_id(1);
    const size_t local_size_x = get_local_size(0);
    const size_t local_size_y = get_local_size(1);

    const size_t group_id_x   = get_group_id(0);
    const size_t group_id_y   = get_group_id(1);
    const size_t num_groups_x = get_num_groups(0);
    const size_t num_groups_y = get_num_groups(1);

    const bool P_W = local_id_x == 0              && group_id_x > 0;
    const bool P_N = local_id_y == 0              && group_id_y > 0;
    const bool P_E = local_id_x == local_size_x-1 && group_id_x < num_groups_x-1;
    const bool P_S = local_id_y == local_size_y-1 && group_id_y < num_groups_y-1;

    uint x = local_id_x + padding;
    uint y = local_id_y + padding;

    // Fill in the halo. Each thread fill in the pixel it's index points to. If
    // it is a border thread, it also has to load the `padding` amount of
    // pixels in its border direction. Corner threads have to fill in pixels
    // in both border directions, as well as the diagonal.
    //
    // Non-corner threads only have to copy `padding+1` number of pixels,
    // and the corner threads have to copy `padding+1`^2. With the filters used
    // in this program, this maxes out at 3 pixels for the non-corner threads,
    // and 9 for the corner threads.

    shared_in[y][x] = in[global_y*width + global_x];
    for (int i = 1; i < padding+1; i++) {
        if (P_W) shared_in[y][x-i] = in[global_y*width     + global_x - i];
        if (P_E) shared_in[y][x+i] = in[global_y*width     + global_x + i];
        if (P_N) shared_in[y-i][x] = in[(global_y-i)*width + global_x];
        if (P_S) shared_in[y+i][x] = in[(global_y+i)*width + global_x];

        /* north west */
        if (P_N && P_W){
            shared_in[y-i][x-i] = in[(global_y-i)*width + global_x - i];
            for (int j = 1; j < i; j++) {
                shared_in[y-i][x-i+j] = in[(global_y-i)*width + global_x - i+j];
                shared_in[y-i+j][x-i] = in[(global_y-i+j)*width + global_x - i];
            }
        }
        /* south west */
        if (P_S && P_W)  {
            shared_in[y+i][x-i] = in[(global_y+i)*width + global_x - i];
            for (int j = 1; j < i; j++) {
                shared_in[y+i][x-i+j] = in[(global_y+i)*width + global_x - i+j];
                shared_in[y+i-j][x-i] = in[(global_y+i-j)*width + global_x - i];
            }
        }
        /* north east */
        if (P_N && P_E) {
            shared_in[y-i][x+i] = in[(global_y-i)*width + global_x + i];
            for (int j = 1; j < i; j++) {
                shared_in[y-i][x+i-j] = in[(global_y-i)*width + global_x + i-j];
                shared_in[y-i+j][x+i] = in[(global_y-i+j)*width + global_x + i];
            }
        }
        /* south east*/
        if (P_S && P_E) {
            shared_in[y+i][x+i] = in[(global_y+i)*width + global_x + i];
            for (int j = 1; j < i; j++) {
                shared_in[y+i][x+i-j] = in[(global_y+i)*width + global_x + i-j];
                shared_in[y+i-j][x+i] = in[(global_y+i-j)*width + global_x + i];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    const uint filterCenter = (filterDim / 2);
    int ar = 0, ag = 0, ab = 0;
    for (uint ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (uint kx = 0; kx < filterDim; kx++) {
            int nkx = filterDim - 1 - kx;

            int yy = y + (ky - filterCenter);
            int xx = x + (kx - filterCenter);
            int global_yy = global_y + (ky - filterCenter);
            int global_xx = global_x + (kx - filterCenter);
            if (global_xx >= 0 && global_xx < (int) width && global_yy >=0 && global_yy < (int) height) {
#ifdef NO_SHARED_MEM
                ar += in[yy*width+xx].r * filter[nky * filterDim + nkx];
                ag += in[yy*width+xx].g * filter[nky * filterDim + nkx];
                ab += in[yy*width+xx].b * filter[nky * filterDim + nkx];

#else
                ar += shared_in[yy][xx].r * filter[nky * filterDim + nkx];
                ag += shared_in[yy][xx].g * filter[nky * filterDim + nkx];
                ab += shared_in[yy][xx].b * filter[nky * filterDim + nkx];
#endif
            }
        }
    }

    ar *= filterFactor;
    ag *= filterFactor;
    ab *= filterFactor;
    ar = (ar < 0) ? 0 : ar;
    ag = (ag < 0) ? 0 : ag;
    ab = (ab < 0) ? 0 : ab;

    out[global_y*width + global_x].r = (ar > 255) ? 255 : ar;
    out[global_y*width + global_x].g = (ag > 255) ? 255 : ag;
    out[global_y*width + global_x].b = (ab > 255) ? 255 : ab;
}
