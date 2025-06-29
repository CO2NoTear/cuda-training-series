#include <stdio.h>

// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

const int DSIZE = 32 * 1048576;
// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds) {

  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < ds;
       idx += gridDim.x * blockDim.x) // a grid-stride loop
    C[idx] = A[idx] + B[idx];         // do the vector (element) add here
}

int main() {

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;
  // start timing
  t0 = clock();

  h_A = new float[DSIZE]; // allocate space for vectors in host memory
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  for (int i = 0; i < DSIZE; i++) { // initialize vectors in host memory
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
    h_C[i] = 0;
  }
  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);
  cudaMalloc(&d_A, DSIZE * sizeof(float)); // allocate device space for vector A
  cudaMalloc(&d_B, DSIZE * sizeof(float)); // allocate device space for vector B
  cudaMalloc(&d_C, DSIZE * sizeof(float)); // allocate device space for vector C
  cudaCheckErrors("cudaMalloc failure");   // error checking
  // copy vector A to device:
  cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  // copy vector B to device:
  cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  // cuda processing sequence step 1 is complete
  // 一共1<<25 = DSIZE = 32 * 1048576个元素
  const int threads_rank = 5;
  const int blocks_rank = 5;
  const int threads = 1 << threads_rank; // modify this line for experimentation
  const int blocks = 1 << blocks_rank;   // modify this line for experimentation
  vadd<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");
  // cuda processing sequence step 2 is complete
  //  copy vector C from device to host:
  cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  // cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  printf("A[0] = %f\n", h_A[0]);
  printf("B[0] = %f\n", h_B[0]);
  printf("C[0] = %f\n", h_C[0]);
  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);
  return 0;
}
