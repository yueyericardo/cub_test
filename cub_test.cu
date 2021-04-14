#include <ATen/cuda/Exceptions.h>
#include <assert.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <stdio.h>

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...)                                                 \
  do {                                                                         \
    size_t temp_storage_bytes = 0;                                             \
    func(nullptr, temp_storage_bytes, __VA_ARGS__);                            \
    auto temp_storage = allocator->allocate(temp_storage_bytes);               \
    func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);                 \
    AT_CUDA_CHECK(cudaGetLastError());                                         \
  } while (false)

template <typename DataT, typename LambdaOpT>
int cubDeviceSelectIf(const DataT *d_in, DataT *d_out, int num_items,
                      LambdaOpT select_op, cudaStream_t stream) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  auto buffer_count = allocator->allocate(sizeof(int));
  int *d_num_selected_out = (int *)buffer_count.get();

  CUB_WRAPPER(cub::DeviceSelect::If, d_in, d_out, d_num_selected_out, num_items,
              select_op, stream);

  // TODO copy num_selected to host, this part is slow
  int num_selected = 0;
  cudaMemcpyAsync(&num_selected, d_num_selected_out, sizeof(int),
                  cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);
  return num_selected;
}

void initData(int *a, int N) {
  for (int i = 0; i < N; ++i) {
    a[i] = i;
  }
}

int main() {
  const int N = 100;
  size_t size = N * sizeof(float);

  int *a;
  int *b;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::globalContext().lazyInitCUDA();

  checkCuda(cudaMallocManaged(&a, size));
  checkCuda(cudaMallocManaged(&b, size));

  initData(a, N);
  int selected = cubDeviceSelectIf(
      a, b, N, [=] __device__(const int i) { return (bool)(i % 2); }, stream);
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());

  printf("selected %d\n", selected);
  assert(selected == N / 2);
  printf("PASS!\n");

  checkCuda(cudaFree(a));
  checkCuda(cudaFree(b));
}
