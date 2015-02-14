#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
		int numAColumns, int numBRows, int numBColumns,
		int numCRows, int numCColumns) {
	int cCol = blockDim.x*blockIdx.x + threadIdx.x;
	int cRow = blockDim.y*blockIdx.y + threadIdx.y;	

	if (cCol < numCColumns && cRow < numCRows) {		
		float cValue = 0;
		for (int i = 0; i < numAColumns; ++i) {
			cValue += A[cRow*numAColumns + i] * B[i*numBColumns + cCol];
		}
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
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows;    // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set this)

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
	hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
	//numAColumns must be == numBRows for valid multiplication. matrix product of [A(n x m)][B(m x p)] = [C(n x p)]
	numCRows = numARows;
	numCColumns = numBColumns;
	hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceA, numARows*numAColumns*sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceB, numBRows*numBColumns*sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceC, numCRows*numCColumns*sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	dim3 dimGrid((numCColumns - 1) / 16 + 1, (numCRows - 1) / 16 + 1, 1);
	dim3 dimBlock(16, 16, 1);
	wbTime_start(Compute, "Performing CUDA computation");
	matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	wbCheck(cudaGetLastError());
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost));
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
