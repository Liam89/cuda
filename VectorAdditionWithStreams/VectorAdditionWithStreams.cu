#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
				        }                                                                     \
			    } while(0)

// Must be at least 2
#define NUM_STREAMS 4

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < len) {
		out[i] = in1[i] + in2[i];
	}
}

int main(int argc, char **argv) {
	wbArg_t args;
	int inputLength;
	float *hostInput1;
	float *hostInput2;
	float *hostOutput;
	float *deviceInput1;
	float *deviceInput2;
	float *deviceOutput;
	cudaStream_t streams[NUM_STREAMS];

	for (int i = 0; i < NUM_STREAMS; ++i){
		cudaStreamCreate(streams + i);
	}

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
	// Use pinned memory
	cudaHostRegister(hostInput1, inputLength, 0);
	cudaHostRegister(hostInput2, inputLength, 0);
	cudaHostAlloc((void **)&hostOutput, inputLength * sizeof(float),0);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInput1, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceInput2, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput, inputLength*sizeof(float)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	// Vector addition with streams
	// For each stream, memory will be copied to the device, the kernal launched, and the memory copied from the device
	// Therefore, the main loop consists of copying memory from the device for stream s-1, launching the kernal for stream s, 
	//	and copying memory to the device for stream s+1
	wbTime_start(Compute, "Performing computation");
	int segSize = (inputLength - 1) / NUM_STREAMS + 1;
	int offset = 0;
	int streamNum = 0;

	// Set up loop for stream 0 and stream 1
	wbCheck(cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, segSize*sizeof(float), cudaMemcpyHostToDevice, streams[streamNum]));
	wbCheck(cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, segSize*sizeof(float), cudaMemcpyHostToDevice, streams[streamNum]));

	dim3 dimGrid((inputLength - 1) / 256 + 1, 1, 1);
	dim3 dimBlock(256, 1, 1);
	vecAdd<<<dimGrid, dimBlock, 0, streams[streamNum]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, segSize);

	streamNum = 1;
	offset += segSize;
	wbCheck(cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, segSize*sizeof(float), cudaMemcpyHostToDevice, streams[streamNum]));
	wbCheck(cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, segSize*sizeof(float), cudaMemcpyHostToDevice, streams[streamNum]));

	// Loop through streams
	for (streamNum = 1; streamNum < NUM_STREAMS - 1; ++streamNum, offset += segSize){
		wbCheck(cudaMemcpyAsync(deviceInput1 + offset + segSize, hostInput1 + offset + segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, streams[streamNum + 1]));
		wbCheck(cudaMemcpyAsync(deviceInput2 + offset + segSize, hostInput2 + offset + segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, streams[streamNum + 1]));

		vecAdd<<<dimGrid, dimBlock, 0, streams[streamNum]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, segSize);

		wbCheck(cudaMemcpyAsync(hostOutput + offset - segSize, deviceOutput + offset - segSize, segSize*sizeof(float), cudaMemcpyDeviceToHost, streams[streamNum - 1]));
	}

	vecAdd<<<dimGrid, dimBlock, 0, streams[streamNum]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, segSize);
	
	wbCheck(cudaMemcpyAsync(hostOutput + offset - segSize, deviceOutput + offset - segSize, segSize*sizeof(float), cudaMemcpyDeviceToHost, streams[streamNum - 1]));

	wbCheck(cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, segSize*sizeof(float), cudaMemcpyDeviceToHost, streams[streamNum]));

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing computation");


	wbTime_start(GPU, "Freeing GPU Memory");
	wbCheck(cudaFree(deviceInput1));
	wbCheck(cudaFree(deviceInput2));
	wbCheck(cudaFree(deviceOutput));
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, inputLength);

	cudaHostUnregister(hostInput1);
	cudaHostUnregister(hostInput2);
	free(hostInput1);
	free(hostInput2);
	cudaFreeHost(hostOutput);

	return 0;
}
