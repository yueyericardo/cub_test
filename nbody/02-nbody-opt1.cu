#include "files.h"
#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct {
  float x, y, z, vx, vy, vz;
} Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */
template <int BLOCKX, int BLOCKY>
__global__ void bodyForce(Body *bodies, float dt, int n) {
  constexpr int BLOCK_SIZE = BLOCKX * BLOCKY;
  __shared__ float3 spos[BLOCK_SIZE];
  int i = blockIdx.x * blockDim.y + threadIdx.y;
  int tIdx = threadIdx.y * blockDim.x + threadIdx.x;
  int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float Fx = 0.0f;
  float Fy = 0.0f;
  float Fz = 0.0f;
  Body bodyI;
  if (i < n)
    bodyI = bodies[i];

  for (int tileIdx = 0; tileIdx < num_tiles; tileIdx++) {
    // preload j to smem
    int j = BLOCK_SIZE * tileIdx + tIdx;
    if (j < n) {
      Body posj = bodies[j];
      spos[tIdx] = make_float3(posj.x, posj.y, posj.z);
    }
    __syncthreads();

    // calculation
    for (int jj = threadIdx.x; jj < BLOCK_SIZE && i < n; jj += blockDim.x) {
      int j = jj + BLOCK_SIZE * tileIdx;
      if (j < n) {
        float dx = spos[jj].x - bodyI.x;
        float dy = spos[jj].y - bodyI.y;
        float dz = spos[jj].z - bodyI.z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
      }
    }
    __syncthreads();
  }

  if (i < n) {
    for (int offset = 16; offset > 0; offset /= 2) {
      Fx += __shfl_down_sync(0xFFFFFFFF, Fx, offset);
      Fy += __shfl_down_sync(0xFFFFFFFF, Fy, offset);
      Fz += __shfl_down_sync(0xFFFFFFFF, Fz, offset);
    }
    if (threadIdx.x == 0) {
      bodies[i].vx += dt * Fx;
      bodies[i].vy += dt * Fy;
      bodies[i].vz += dt * Fz;
    }
  }
}

__global__ void integratePos(Body *bodies, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n)
    return;
  // printf("cuda %d %f \n", i, bodies[i].vx);
  bodies[i].x += bodies[i].vx * dt;
  bodies[i].y += bodies[i].vy * dt;
  bodies[i].z += bodies[i].vz * dt;
}

int main(const int argc, const char **argv) {

  // The assessment will test against both 2<11 and 2<15.
  // Feel free to pass the command line argument 15 when you gernate ./nbody
  // report files
  int nBodies = 2 << 11;
  if (argc > 1)
    nBodies = 2 << atoi(argv[1]);

  // The assessment will pass hidden initialized values to check for
  // correctness. You should not make changes to these files, or else the
  // assessment will not work.
  const char *initialized_values;
  const char *solution_values;

  if (nBodies == 2 << 11) {
    initialized_values = "files/initialized_4096";
    solution_values = "files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "files/initialized_65536";
    solution_values = "files/solution_65536";
  }

  if (argc > 2)
    initialized_values = argv[2];
  if (argc > 3)
    solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  int deviceId;
  cudaGetDevice(&deviceId);

  cudaMallocManaged(&buf, bytes);
  Body *p = (Body *)buf;

  cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId);
  read_values_from_file(initialized_values, buf, bytes);
  cudaMemPrefetchAsync(buf, bytes, deviceId);

  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

    /*
     * You will likely wish to refactor the work being done in `bodyForce`,
     * and potentially the work to integrate the positions.
     */
    constexpr int iPerblock = 16;
    dim3 block(32, iPerblock);
    int nblocks = (nBodies + iPerblock - 1) / iPerblock;
    bodyForce<32, iPerblock><<<nblocks, block>>>(p, dt, nBodies);

    /*
     * This position integration cannot occur until this round of `bodyForce`
     * has completed. Also, the next round of `bodyForce` cannot begin until the
     * integration is complete.
     */

    int blocksize = 128;
    nblocks = (nBodies + blocksize - 1) / blocksize;
    integratePos<<<nblocks, blocksize>>>(p, dt, nBodies);
    cudaDeviceSynchronize();

    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  write_values_to_file(solution_values, buf, bytes);

  // You will likely enjoy watching this value grow as you accelerate the
  // application, but beware that a failure to correctly synchronize the device
  // might result in unrealistically high values.
  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

  cudaFree(buf);
}
