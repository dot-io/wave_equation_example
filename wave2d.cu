#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static inline void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ __forceinline__ int mirror_idx(int i, int n) {
    // Mirror boundary for one-step-out neighbor access.
    // For stencil accesses we only ever ask for i in {-1..n}, so this is enough.
    if (i < 0)      return -i;              // -1 -> 1
    if (i >= n)     return 2 * n - 2 - i;   // n -> n-2
    return i;
}

__global__ void init_gaussian(float* cur, float* last, int nx, int ny,
                              float h, float x0, float y0, float sigma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y
    if (i >= nx || j >= ny) return;

    float x = i * h;
    float y = j * h;
    float dx = x - x0;
    float dy = y - y0;
    float g = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));

    int idx = j * nx + i;
    cur[idx]  = g;

    // Common choice: zero initial velocity => u^{n-1} = u^{n} at t=0
    // (More accurate start-ups exist, but this is typical and simple.)
    last[idx] = g;
}

__global__ void wave_step_shared(const float* __restrict__ last,
                                 const float* __restrict__ cur,
                                 float* __restrict__ next,
                                 int nx, int ny, float r2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    extern __shared__ float sh[];

    int sx = threadIdx.x + 1;
    int sy = threadIdx.y + 1;

    int stride = blockDim.x + 2;

    sh[sy * stride + sx] = cur[j * nx + i];

    if (threadIdx.x == 0) {
        int im = mirror_idx(i - 1, nx);
        sh[sy * stride + 0] = cur[j * nx + im];
    }
    if (threadIdx.x == blockDim.x - 1) {
        int ip = mirror_idx(i + 1, nx);
        sh[sy * stride + (blockDim.x + 1)] = cur[j * nx + ip];
    }
    if (threadIdx.y == 0) {
        int jm = mirror_idx(j - 1, ny);
        sh[0 * stride + sx] = cur[jm * nx + i];
    }
    if (threadIdx.y == blockDim.y - 1) {
        int jp = mirror_idx(j + 1, ny);
        sh[(blockDim.y + 1) * stride + sx] = cur[jp * nx + i];
    }

    __syncthreads();

    float uC = sh[sy * stride + sx];
    float lap = sh[sy * stride + (sx - 1)] + sh[sy * stride + (sx + 1)] +
                sh[(sy - 1) * stride + sx] + sh[(sy + 1) * stride + sx] - 4.0f * uC;

    float uLast = last[j * nx + i];
    next[j * nx + i] = 2.0f * uC - uLast + r2 * lap;
}

int main() {
    // Domain
    int nx = 512, ny = 512;
    float h  = 1.0f;
    float c  = 1.0f;
    float dt = 0.5f; // choose dt so stable; see note below

    float r  = (c * dt / h);
    float r2 = r * r;

    int steps = 500;

    size_t bytes = (size_t)nx * (size_t)ny * sizeof(float);

    float *d_last = nullptr, *d_cur = nullptr, *d_next = nullptr;
    check(cudaMalloc(&d_last, bytes), "cudaMalloc d_last");
    check(cudaMalloc(&d_cur,  bytes), "cudaMalloc d_cur");
    check(cudaMalloc(&d_next, bytes), "cudaMalloc d_next");

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Gaussian initial condition
    float x0 = 0.5f * (nx - 1) * h;
    float y0 = 0.5f * (ny - 1) * h;
    float sigma = 20.0f * h;

    init_gaussian<<<grid, block>>>(d_cur, d_last, nx, ny, h, x0, y0, sigma);
    check(cudaGetLastError(), "init_gaussian launch");
    check(cudaDeviceSynchronize(), "init_gaussian sync");

    // Shared memory size for tile+halo
    size_t shBytes = (size_t)(block.x + 2) * (size_t)(block.y + 2) * sizeof(float);

    for (int n = 0; n < steps; ++n) {
        wave_step_shared<<<grid, block, shBytes>>>(d_last, d_cur, d_next, nx, ny, r2);
        check(cudaGetLastError(), "wave_step_shared launch");

        // Rotate pointers: last <- cur, cur <- next, next <- last
        float* tmp = d_last;
        d_last = d_cur;
        d_cur = d_next;
        d_next = tmp;
    }

    check(cudaDeviceSynchronize(), "final sync");

    // (Optional) copy back one value just to show it's alive
    float sample = 0.0f;
    check(cudaMemcpy(&sample, d_cur + (ny/2)*nx + (nx/2), sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy sample");
    std::printf("Center sample after %d steps: %g\n", steps, sample);

    cudaFree(d_last);
    cudaFree(d_cur);
    cudaFree(d_next);
    return 0;
}
