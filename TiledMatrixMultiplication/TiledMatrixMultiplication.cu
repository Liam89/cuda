#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_WIDTH 4

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
	    }                                                                          \
    } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
		int numAColumns, int numBRows,
		int numBColumns, int numCRows,
		int numCColumns) {

	__shared__ float sharedA[TILE_WIDTH * TILE_WIDTH];
	__shared__ float sharedB[TILE_WIDTH * TILE_WIDTH];

	int cCol = blockDim.x*blockIdx.x + threadIdx.x;
	int cRow = blockDim.y*blockIdx.y + threadIdx.y;
	int ax;
	int by;
	float cValue = 0;
	int numTiles = (numAColumns - 1) / TILE_WIDTH + 1;

	// Loop through tiles
	for (int tile = 0; tile < numTiles; ++tile) {
		// Each thread loads the matrix element from A and B, into shared tile memory, corresponding to its relative position
		ax = tile*TILE_WIDTH + threadIdx.x;
		by = tile*TILE_WIDTH + threadIdx.y;
		if (ax < numAColumns && cRow < numARows) {
			sharedA[TILE_WIDTH*threadIdx.y + threadIdx.x] = A[cRow*numAColumns + ax];
		} else {
			sharedA[TILE_WIDTH*threadIdx.y + threadIdx.x] = 0.0;
		}

		if (cCol < numBColumns && by < numBRows) {
			sharedB[TILE_WIDTH*threadIdx.y + threadIdx.x] = B[by*numBColumns + cCol];
		} else {
			sharedB[TILE_WIDTH*threadIdx.y + threadIdx.x] = 0.0;
		}
		__syncthreads();

		// Calculate the product of the tile slices
		if (cCol < numCColumns && cRow < numCRows) {
			for (int i = 0; i < TILE_WIDTH; ++i) {
				cValue += sharedA[threadIdx.y*TILE_WIDTH + i] * sharedB[i*TILE_WIDTH + threadIdx.x];
			}
		}
		__syncthreads();
	}
	if (cCol < numCColumns && cRow < numCRows) {
		C[cRow*numCColumns + cCol] = cValue;
	}

}

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostA; // The A matrix
	float *hostB; // The B matrix
	float *hostC; // The output C matrix
	float *deviceA;
	float *deviceB;
	float *deviceC;
	int numARows;    // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int sizeAMem;
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int sizeBMem;
	int numCRows;    // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set this)
	int sizeCMem;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
	hostB =	(float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
	sizeAMem = numARows*numAColumns*sizeof(float);
	sizeBMem = numBRows*numBColumns*sizeof(float);

	numCRows = numARows;
	numCColumns = numBColumns;
	sizeCMem = numCRows*numCColumns*sizeof(float);
	hostC = (float *)malloc(sizeCMem);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceA, sizeAMem));
	wbCheck(cudaMalloc((void **)&deviceB, sizeBMem));
	wbCheck(cudaMalloc((void **)&deviceC, sizeCMem));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceA, hostA, sizeAMem, cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceB, hostB, sizeBMem, cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 dimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
	wbTime_start(Compute, "Performing CUDA computation");
	matrixMultiplyShared<<<dimGrid,dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostC, deviceC, sizeCMem, cudaMemcpyDeviceToHost));
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	wbCheck(cudaFree(deviceA));
	wbCheck(cudaFree(deviceB));
	wbCheck(cudaFree(deviceC));
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostC, numCRows, numCColumns);

	free(hostA);
	free(hostB);
	free(hostC);

	return 0;
}
