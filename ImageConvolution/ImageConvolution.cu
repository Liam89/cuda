#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_IMAGE_WIDTH 12
// Calculation tile width. Mask size fixed at 5
#define TILE_TOTAL_WIDTH (TILE_IMAGE_WIDTH+4)

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
		        }                                                                     \
	    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

__device__ float clamp(float x, float start, float end)
{
	return min(max(x, start), end);
}

__global__ void imageConvolution(float * inputImage, const float * __restrict__ mask, // Use constant memory
	float * outputImage, int imageWidth, int imageHeight, int imageChannels, int maskRows, int maskColumns) {

	__shared__ float tileImage[TILE_TOTAL_WIDTH * TILE_TOTAL_WIDTH * 3]; // @todo use imageChannels
	// Offset to top left corner of tile
	int cornerOffsetX = (TILE_IMAGE_WIDTH*blockIdx.x - Mask_radius);
	int cornerOffsetY = (TILE_IMAGE_WIDTH*blockIdx.y - Mask_radius);

	// load calculation tile. Linearized then mapped to (c,x,y) for sequential memory access.
	for (int c = 0; c < imageChannels; ++c) {
		int loadIndex = threadIdx.x + blockDim.x*threadIdx.y + c*blockDim.x*blockDim.y;
		
		// Map the tile position from (x,y,c) to (c,x,y)
		int tileIdxC = loadIndex % imageChannels;
		int tileIdxXY = loadIndex / imageChannels;
		int tileIdxX = tileIdxXY % TILE_TOTAL_WIDTH;
		int tileIdxY = tileIdxXY / TILE_TOTAL_WIDTH;
		// Calculate the position to load from the input image
		int inIdxC = tileIdxC;
		int inIdxX = tileIdxX + cornerOffsetX;
		int inIdxY = tileIdxY + cornerOffsetY;
		// Load memory if in valid range, otherwise load 0.0;
		if (inIdxX >= 0 && inIdxY >= 0 && inIdxX < imageWidth && inIdxY < imageHeight) {
			tileImage[loadIndex] = inputImage[(imageWidth*inIdxY + inIdxX)*imageChannels + inIdxC];
		}
		else {
			tileImage[loadIndex] = 0.0;
		}
	}

	__syncthreads();

	// Apply mask to tile
	// mask center, relative to top left corner of tile
	int mcx = threadIdx.x;
	int mcy = threadIdx.y;
	// Only calculate for threads within the TILE_IMAGE_WIDTH sub region of the tile and if within total image height and width
	if (mcx >= Mask_radius && mcx < (TILE_IMAGE_WIDTH + Mask_radius) && mcy >= Mask_radius && mcy < (TILE_IMAGE_WIDTH + Mask_radius)
		&& (cornerOffsetX + mcx) < imageWidth && (cornerOffsetY + mcy) < imageHeight) {
		for (int c = 0; c < imageChannels; ++c){
			float oValue = 0.0;		
			for (int y = 0; y < Mask_width; ++y) {
				for (int x = 0; x < Mask_width; ++x) {
					int tileImageIndex = ((mcy - Mask_radius + y)*TILE_TOTAL_WIDTH + (mcx - Mask_radius + x))*imageChannels + c;
					int maskIndex = (y*Mask_width + x);
					oValue += tileImage[tileImageIndex] * mask[maskIndex];
				}
			}
			int outputIndex = ((cornerOffsetY + mcy)*imageWidth + (cornerOffsetX + mcx))*imageChannels + c;
			outputImage[outputIndex] = clamp(oValue, 0.0, 1.0);
		}
	}
}


int main(int argc, char* argv[]) {
	wbArg_t args;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char * inputImageFile;
	char * inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float * hostInputImageData;
	float * hostOutputImageData;
	float * hostMaskData;
	float * deviceInputImageData;
	float * deviceOutputImageData;
	float * deviceMaskData;

	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);
	inputMaskFile = wbArg_getInputFile(args, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");


	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData,
		hostInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData,
		hostMaskData,
		maskRows * maskColumns * sizeof(float),
		cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");


	dim3 dimBlock(TILE_TOTAL_WIDTH, TILE_TOTAL_WIDTH, 1);
	dim3 dimGrid((imageWidth - 1) / TILE_IMAGE_WIDTH + 1, (imageHeight - 1) / TILE_IMAGE_WIDTH + 1, 1);
	wbTime_start(Compute, "Doing the computation on the GPU");
	imageConvolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageWidth, imageHeight, imageChannels, maskRows, maskColumns);
	cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
	wbTime_stop(Compute, "Doing the computation on the GPU");


	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData,
		deviceOutputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
