#include <cstdio>

__global__ void hello_kernel() {
    // add hello info and print block
    printf("Hello from block %d , thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    printf("Hello from the host!\n");
    return 0;
}
