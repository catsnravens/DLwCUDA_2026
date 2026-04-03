#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define WIDTH  800
#define HEIGHT 1200
#define ITERS  100000
#define THREADS 256
#define BLOCKS  256

// Simple linear congruential generator for per-thread random numbers.
// Each thread gets a different seed, so trajectories diverge.
__device__ unsigned int lcg_rand(unsigned int *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

// Barnsley fern IFS (Iterated Function System).
// Picks one of 4 affine transforms based on random probability:
//   f1 (1%):  stem
//   f2 (85%): successively smaller leaflets
//   f3 (7%):  largest left leaflet
//   f4 (7%):  largest right leaflet
//
// Implement the 4 affine transformations.
// Use (r % 1000) / 1000.0f as a probability in [0, 1).
// The coefficients are:
//   f1: nx = 0,                         ny = 0.16 * y
//   f2: nx = 0.85*x + 0.04*y,           ny = -0.04*x + 0.85*y + 1.6
//   f3: nx = 0.20*x - 0.26*y,           ny = 0.23*x + 0.22*y + 1.6
//   f4: nx = -0.15*x + 0.28*y,          ny = 0.26*x + 0.24*y + 0.44
__device__ void fern_transform(float x, float y, float *nx, float *ny, unsigned int r) {
    float p = (r % 1000) / 1000.0f;
    if (p < 0.01f) {
        // f1: stem
        *nx = 0.0f;
        *ny = 0.16f * y;
    } else if (p < 0.86f) {
        // f2: successively smaller leaflets
        *nx = 0.85f * x + 0.04f * y;
        *ny = -0.04f * x + 0.85f * y + 1.6f;
    } else if (p < 0.93f) {
        // f3: largest left leaflet
        *nx = 0.20f * x - 0.26f * y;
        *ny = 0.23f * x + 0.22f * y + 1.6f;
    } else {
        // f4: largest right leaflet
        *nx = -0.15f * x + 0.28f * y;
        *ny = 0.26f * x + 0.24f * y + 0.44f;
    }
}

// TODO: Implement the fern kernel.
// Each thread should:
//   1. Compute a unique thread ID and seed the RNG with it
//   2. Start at (x, y) = (0, 0)
//   3. For `iters` iterations:
//      a. Generate a random number with lcg_rand
//      b. Apply fern_transform to get new (x, y)
//      c. Map (x, y) to pixel coordinates:
//         px = (int)((x + 2.182) / 4.84 * width)
//         py = (int)((9.998 - y) / 9.998 * height)
//      d. If (px, py) is in bounds, increment histogram[py * width + px]
//         using atomicAdd
__global__ void fern_kernel(int *histogram, int width, int height, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int state = tid; // Seed RNG with unique thread ID

    float x = 0.0f, y = 0.0f;
    for (int i = 0; i < iters; i++) {
        unsigned int r = lcg_rand(&state);
        float nx, ny;
        fern_transform(x, y, &nx, &ny, r);
        x = nx;
        y = ny;

        int px = (int)((x + 2.182f) / 4.84f * width);
        int py = (int)((9.998f - y) / 9.998f * height);
        if (px >= 0 && px < width && py >= 0 && py < height) {
            atomicAdd(&histogram[py * width + px], 1);
        }
    }

}

void save_bmp(const char *filename, const int *hist, int width, int height) {
    int max_val = 0;
    for (int i = 0; i < width * height; i++)
        if (hist[i] > max_val) max_val = hist[i];

    int row_size = (width * 3 + 3) & ~3;
    int pixel_data_size = row_size * height;
    int file_size = 54 + pixel_data_size;

    unsigned char header[54] = {};
    header[0] = 'B'; header[1] = 'M';
    header[2] = file_size; header[3] = file_size >> 8;
    header[4] = file_size >> 16; header[5] = file_size >> 24;
    header[10] = 54;
    header[14] = 40;
    header[18] = width; header[19] = width >> 8;
    header[22] = height; header[23] = height >> 8;
    header[26] = 1;
    header[28] = 24;

    FILE *f = fopen(filename, "wb");
    fwrite(header, 1, 54, f);

    unsigned char *row = (unsigned char *)calloc(row_size, 1);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            float t = max_val > 0 ? (float)hist[y * width + x] / max_val : 0.0f;
            t = logf(1.0f + t * 99.0f) / logf(100.0f);
            row[x * 3 + 0] = (unsigned char)(t * 30);
            row[x * 3 + 1] = (unsigned char)(t * 255);
            row[x * 3 + 2] = (unsigned char)(t * 20);
        }
        fwrite(row, 1, row_size, f);
    }
    free(row);
    fclose(f);
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(int);

    // Allocate GPU memory for the histogram and zero it out
    int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_hist, size));
    CUDA_CHECK(cudaMemset(d_hist, 0, size));

    // Launch the fern_kernel
    fern_kernel<<<BLOCKS, THREADS>>>(d_hist, WIDTH, HEIGHT, ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back and save
    int *h_hist = (int *)malloc(size);
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, size, cudaMemcpyDeviceToHost));

    save_bmp("fern.bmp", h_hist, WIDTH, HEIGHT);
    printf("Saved fern.bmp (%dx%d)\n", WIDTH, HEIGHT);

    CUDA_CHECK(cudaFree(d_hist));
    free(h_hist);
    return 0;
}
