#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

static void check_cuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

__device__ __forceinline__ int mirror_idx(int i, int n) {
    if (i < 0)  return -i;              // -1 -> 1
    if (i >= n) return 2 * n - 2 - i;   // n -> n-2, n+1 -> n-3, ...
    return i;
}

__global__ void init_gaussian(float* cur, float* last,
                              int nx, int ny,
                              float h, float x0, float y0, float sigma)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= nx || j >= ny) return;

    float x = i * h;
    float y = j * h;
    float dx = x - x0;
    float dy = y - y0;
    float g  = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));

    int id = j * nx + i;
    cur[id]  = g;
    last[id] = g; // zero initial velocity: u^{n-1} = u^n
}

#define SH(sh, pitch, x, y) (sh[(y) * (pitch) + (x)])

__global__ void wave_steps_coop(const float* __restrict__ last0,
                               const float* __restrict__ cur0,
                               float* __restrict__ next0,
                               int nx, int ny, float r2, int steps)
{
    // Cooperative "grid group" for grid-wide synchronization
    cg::grid_group grid = cg::this_grid();

    // Local pointer rotation (in registers)
    const float* last = last0;
    const float* cur  = cur0;
    float* next       = next0;

    // Block / thread info
    int Bx = (int)blockDim.x;
    int By = (int)blockDim.y;
    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;

    // Shared tile with 1-cell halo: (Bx+2) x (By+2)
    extern __shared__ float sh[];
    int sPitch = Bx + 2;
    int sx = tx + 1;
    int sy = ty + 1;

    // Number of tiles covering the domain
    int tilesX = (nx + Bx - 1) / Bx;
    int tilesY = (ny + By - 1) / By;

    for (int n = 0; n < steps; ++n) {

        // Each cooperative block may cover multiple tiles via grid-stride loops
        for (int tileY = (int)blockIdx.y; tileY < tilesY; tileY += (int)gridDim.y) {
            for (int tileX = (int)blockIdx.x; tileX < tilesX; tileX += (int)gridDim.x) {

                // Global indices for this tile and thread
                int i0 = tileX * Bx + tx;
                int j0 = tileY * By + ty;

                // For shared loads, always map to a valid (mirrored) global index
                int iC = mirror_idx(i0, nx);
                int jC = mirror_idx(j0, ny);

                // Load center
                SH(sh, sPitch, sx, sy) = cur[jC * nx + iC];

                // Load halo (mirror boundary). Use mirrored j as needed too.
                if (tx == 0) {
                    int im = mirror_idx(i0 - 1, nx);
                    SH(sh, sPitch, 0, sy) = cur[jC * nx + im];
                }
                if (tx == Bx - 1) {
                    int ip = mirror_idx(i0 + 1, nx);
                    SH(sh, sPitch, Bx + 1, sy) = cur[jC * nx + ip];
                }
                if (ty == 0) {
                    int jm = mirror_idx(j0 - 1, ny);
                    SH(sh, sPitch, sx, 0) = cur[jm * nx + iC];
                }
                if (ty == By - 1) {
                    int jp = mirror_idx(j0 + 1, ny);
                    SH(sh, sPitch, sx, By + 1) = cur[jp * nx + iC];
                }

                __syncthreads(); // ensure full tile+halo is ready before any reads

                // Compute 5-point Laplacian from shared memory
                float uC  = SH(sh, sPitch, sx, sy);
                float lap = SH(sh, sPitch, sx - 1, sy)
                          + SH(sh, sPitch, sx + 1, sy)
                          + SH(sh, sPitch, sx, sy - 1)
                          + SH(sh, sPitch, sx, sy + 1)
                          - 4.0f * uC;

                // Write only for in-domain points (not for out-of-range threads in edge tiles)
                if (i0 < nx && j0 < ny) {
                    int id = j0 * nx + i0;
                    float uLast = last[id];
                    next[id] = 2.0f * uC - uLast + r2 * lap;
                }

                __syncthreads(); // protect shared memory reuse before the next tile iteration
            }
        }

        // Global barrier: all blocks must finish writing next before any block swaps/reads for next step
        grid.sync();

        // Rotate buffers locally in every thread
        {
            const float* tmp = last;
            last = cur;
            cur  = (const float*)next;
            next = (float*)tmp; // safe: we're just rotating base addresses, storage exists
        }

        // Global barrier: ensure no block starts reading from "cur" while some other block is still using old mapping
        grid.sync();
    }
}

int main(void) {
    int nx = 512, ny = 512;
    float h  = 1.0f;
    float c  = 1.0f;
    float dt = 0.5f;
    float r  = (c * dt / h);
    float r2 = r * r;
    int steps = 500;

    int dev = 0;
    check_cuda(cudaSetDevice(dev), "cudaSetDevice");

    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");

    if (!prop.cooperativeLaunch) {
        fprintf(stderr, "Device does not support cooperativeLaunch.\n");
        return 1;
    }

    size_t bytes = (size_t)nx * (size_t)ny * sizeof(float);

    float *d_last = NULL, *d_cur = NULL, *d_next = NULL;
    check_cuda(cudaMalloc((void**)&d_last, bytes), "cudaMalloc d_last");
    check_cuda(cudaMalloc((void**)&d_cur,  bytes), "cudaMalloc d_cur");
    check_cuda(cudaMalloc((void**)&d_next, bytes), "cudaMalloc d_next");

    dim3 block(16, 16);

    // Init uses a normal launch over the full domain
    dim3 gridInit((unsigned)((nx + block.x - 1) / block.x),
                  (unsigned)((ny + block.y - 1) / block.y));

    float x0 = 0.5f * (nx - 1) * h;
    float y0 = 0.5f * (ny - 1) * h;
    float sigma = 20.0f * h;

    init_gaussian<<<gridInit, block>>>(d_cur, d_last, nx, ny, h, x0, y0, sigma);
    check_cuda(cudaGetLastError(), "init_gaussian launch");
    check_cuda(cudaDeviceSynchronize(), "init_gaussian sync");

    // Dynamic shared memory size
    size_t shBytes = (size_t)(block.x + 2) * (size_t)(block.y + 2) * sizeof(float);

    // Compute cooperative grid limits (max blocks that can be resident concurrently)
    int numSM = prop.multiProcessorCount;
    int activeBlocksPerSM = 0;
    int threadsPerBlock = (int)(block.x * block.y);

    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &activeBlocksPerSM,
                   wave_steps_coop,
                   threadsPerBlock,
                   shBytes),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

    int coopMaxBlocksTotal = numSM * activeBlocksPerSM;
    if (coopMaxBlocksTotal <= 0) {
        fprintf(stderr, "Cooperative launch not feasible with this block size/shared usage.\n");
        return 1;
    }

    // Desired number of tiles
    int tilesX = (nx + (int)block.x - 1) / (int)block.x;
    int tilesY = (ny + (int)block.y - 1) / (int)block.y;
    int desiredBlocksTotal = tilesX * tilesY;

    // Choose cooperative grid dims not exceeding coopMaxBlocksTotal.
    // We still cover the whole domain via grid-stride loops inside the kernel.
    int gx = tilesX;
    if (gx > coopMaxBlocksTotal) gx = coopMaxBlocksTotal;
    if (gx < 1) gx = 1;

    int gy = (coopMaxBlocksTotal / gx);
    if (gy < 1) gy = 1;
    if (gy > tilesY) gy = tilesY;

    dim3 gridCoop((unsigned)gx, (unsigned)gy);

    if ((int)(gridCoop.x * gridCoop.y) > coopMaxBlocksTotal) {
        fprintf(stderr, "Internal error: gridCoop exceeds cooperative limit.\n");
        return 1;
    }

    // Cooperative kernel launch
    void* args[] = {
        (void*)&d_last,
        (void*)&d_cur,
        (void*)&d_next,
        (void*)&nx,
        (void*)&ny,
        (void*)&r2,
        (void*)&steps
    };

    check_cuda(cudaLaunchCooperativeKernel(
                   (void*)wave_steps_coop,
                   gridCoop,
                   block,
                   args,
                   shBytes,
                   0),
               "cudaLaunchCooperativeKernel");

    check_cuda(cudaDeviceSynchronize(), "wave_steps_coop sync");

    // Sample center
    {
        float sample = 0.0f;
        int cx = nx / 2, cy = ny / 2;
        check_cuda(cudaMemcpy(&sample,
                              d_cur + (size_t)cy * (size_t)nx + (size_t)cx,
                              sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy sample");
        printf("Center sample after %d steps: %g\n", steps, sample);
        printf("Init blocks: %d, desired blocks: %d, coop grid: %u x %u (total %u), coop max total: %d\n",
               (int)(gridInit.x * gridInit.y), desiredBlocksTotal,
               gridCoop.x, gridCoop.y, (unsigned)(gridCoop.x * gridCoop.y),
               coopMaxBlocksTotal);
    }

    cudaFree(d_last);
    cudaFree(d_cur);
    cudaFree(d_next);
    return 0;
}
