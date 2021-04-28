#include "files.h"
#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SOFTENING 1e-9f
#define BLOCK_SIZE 32
#define BLOCK_STRIDE 32

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

__global__ void bodyForce(Body *p, float dt, int n) {

  // int i = threadIdx.x + blockIdx.x * blockDim.x;
  int cycle_times = n / BLOCK_SIZE;
  // 计算要处理的数据index
  int i = threadIdx.x + (int)(blockIdx.x / BLOCK_STRIDE) * blockDim.x;
  // 此块对应要处理的数据块的起始位置
  int start_block = blockIdx.x % BLOCK_STRIDE;
  if (i < n) {
    Body ptemp = p[i];
    Body temp;
    float share_x, share_y, share_z;
    float dx, dy, dz, distSqr, invDist, invDist3;
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;
    // 这里的cycle_times 在已知块大小时使用常数性能会高一些
    for (int block_num = start_block; block_num < cycle_times;
         block_num += BLOCK_STRIDE) {
      temp = p[block_num * BLOCK_SIZE + threadIdx.x];
      share_x = temp.x;
      share_y = temp.y;
      share_z = temp.z;
      // 编译优化，只有 BLOCK_SIZE 为常量时才有用
#pragma unroll
      for (int j = 0; j < BLOCK_SIZE; j++) {
        dx = __shfl_sync(0xFFFFFFFF, share_x, j) - ptemp.x;
        dy = __shfl_sync(0xFFFFFFFF, share_y, j) - ptemp.y;
        dz = __shfl_sync(0xFFFFFFFF, share_z, j) - ptemp.z;
        distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        invDist = rsqrtf(distSqr);
        invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
      }
      // 块内同步，防止spos提前被写入
      __syncthreads();
    }
    // 块之间不同步，原子加保证正确性
    atomicAdd(&p[i].vx, dt * Fx);
    atomicAdd(&p[i].vy, dt * Fy);
    atomicAdd(&p[i].vz, dt * Fz);
    // p[i].vx += dt * Fx;
    // p[i].vy += dt * Fy;
    // p[i].vz += dt * Fz;
  }
}

__global__ void integratePos(Body *p, float dt, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
  }
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
  cudaDeviceProp props;

  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&props, deviceId);

  dim3 threads_per_block(props.warpSize * 1);
  int nblocks = (nBodies + threads_per_block.x - 1) / threads_per_block.x;

  printf("blocks %d, SMs: %d, threads_per_block %d\n", nblocks,
         props.multiProcessorCount, threads_per_block.x);

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
    cudaDeviceSynchronize();
    bodyForce<<<nblocks * BLOCK_STRIDE, threads_per_block>>>(p, dt, nBodies);

    /*
     * This position integration cannot occur until this round of `bodyForce`
     * has completed. Also, the next round of `bodyForce` cannot begin until the
     * integration is complete.
     */

    integratePos<<<nblocks, threads_per_block>>>(p, dt, nBodies);
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
