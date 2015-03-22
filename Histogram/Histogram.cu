#include <wb.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HISTOGRAM_LENGTH 256
#define TILE_WIDTH 16

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
		    }                                                                          \
      } while (0)


__global__ void castToUChar(float * inputImage, unsigned char * ucharImage, int imageSize) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < imageSize) {
		ucharImage[index] = (unsigned char)(255 * inputImage[index]);
	}
}

__global__ void convertToGreyscale(unsigned char * ucharImage, unsigned char * greyImage, int imageWidth, int imageHeight) {
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	if (col < imageWidth && row < imageHeight) {
		int index = imageWidth*row + col;
		unsigned char r = ucharImage[3 * index];
		unsigned char g = ucharImage[3 * index + 1];
		unsigned char b = ucharImage[3 * index + 2];
		greyImage[index] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
	}
}

__global__ void calcHistogram(unsigned char * greyImage, unsigned int * histogram, long imageSize) {
	__shared__ unsigned int localHistogram[HISTOGRAM_LENGTH];
	// Initalize to zero
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (threadIdx.x < HISTOGRAM_LENGTH) {
		localHistogram[threadIdx.x] = 0;
	}
	__syncthreads();

	
	int stride = blockDim.x * gridDim.x;
	while (i < imageSize) {
		atomicAdd(&(localHistogram[greyImage[i]]), 1);
		i += stride;
	}

	// wait for all other threads in the block to finish
	__syncthreads();
	if (threadIdx.x < HISTOGRAM_LENGTH) {
		atomicAdd(&(histogram[threadIdx.x]), localHistogram[threadIdx.x]);
	}
}

__device__ float normalize(int x, float normConstant) {
	return normConstant*x;
}

//Cumulative Distribution Function of histogram
////Block size must be HISTOGRAM_LENGTH and grid size must be 1
__global__ void calcCDF(unsigned int * histogram, float * cdf, float normConstant) {
	__shared__ float localHistogram[HISTOGRAM_LENGTH];

	localHistogram[threadIdx.x] = histogram[threadIdx.x];
	__syncthreads();

	int sum = 0;
	for (int i = 0; i <= threadIdx.x; ++i) {
		sum += localHistogram[i];
	}
	cdf[threadIdx.x] = normalize(sum, normConstant);
}

//Block size must be HISTOGRAM_LENGTH/2 and grid size must be 1
__global__ void minimum(float * cdf, float * result) {
	__shared__ float partialMin[HISTOGRAM_LENGTH];
	int loadIndex;
	for (int i = 0; i < 2; ++i) {
		loadIndex = 2 * blockIdx.x*blockDim.x + i*blockDim.x + threadIdx.x;
		if (loadIndex < HISTOGRAM_LENGTH) {
			partialMin[i*blockDim.x + threadIdx.x] = cdf[loadIndex];
		}
		else {
			partialMin[i*blockDim.x + threadIdx.x] = cdf[0];
		}
	}

	//Traverse the reduction tree
	int t = threadIdx.x;
	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
		__syncthreads();
		if (t < stride) {
			partialMin[t] = min(partialMin[t], partialMin[t + stride]);
		}
	}
	__syncthreads();
	if (t == 0) {
		*result = partialMin[0];
	}
}

__device__ float clamp(float x, float start, float end) {
	return min(max(x, start), end);
}

__device__ unsigned char correct_colour(int val, float * cdf, float * cdfmin) {
	return (unsigned char)clamp(255 * (cdf[val] - cdfmin[0]) / (1 - cdfmin[0]), 0, 255);
}

__global__ void equalizeImage(unsigned char * ucharImage, float * cdf, float * cdfmin, int imageSize) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < imageSize) {
		ucharImage[index] = correct_colour(ucharImage[index], cdf, cdfmin);
	}
}

__global__ void castToFloat(float * inputImage, unsigned char * ucharImage, int imageSize) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < imageSize) {
		inputImage[index] = (float)(ucharImage[index] / 255.0);
	}
}



int main(int argc, char ** argv) {
	wbArg_t args;
	int imageWidth;
	int imageHeight;
	int imageChannels;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float * hostInputImageData;
	float * hostOutputImageData;
	const char * inputImageFile;
	float * deviceInputImageData;
	unsigned char * deviceGreyImage;
	unsigned char * deviceUCharImage;
	unsigned int * deviceHistogram;
	float * deviceCDF;
	float * deviceCDFMin;


	args = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(args, 0);

	wbTime_start(Generic, "Importing data and creating memory on host");
	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceUCharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
	cudaMalloc((void **)&deviceGreyImage, imageWidth * imageHeight * sizeof(unsigned char));
	cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
	cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
	cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
	cudaMalloc((void **)&deviceCDFMin, sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(
		deviceInputImageData,
		hostInputImageData, 
		imageWidth * imageHeight * imageChannels * sizeof(float), 
		cudaMemcpyHostToDevice
	));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	dim3 dimBlock(TILE_WIDTH*TILE_WIDTH, 1, 1);
	dim3 dimGrid((imageWidth*imageHeight*imageChannels - 1) / (TILE_WIDTH*TILE_WIDTH) + 1, 1, 1);
	wbTime_start(Compute, "Converting image input to uchar");
	castToUChar<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceUCharImage, imageWidth*imageHeight*imageChannels);
	cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
	wbTime_stop(Compute, "Converting image input to uchar");

	dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
	dimGrid = dim3((imageWidth - 1) / TILE_WIDTH + 1, (imageHeight - 1) / TILE_WIDTH + 1, 1);
	wbTime_start(Compute, "Converting to greyscale");
	convertToGreyscale<<<dimGrid, dimBlock>>>(deviceUCharImage, deviceGreyImage, imageWidth, imageHeight);
	cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
	wbTime_stop(Compute, "Converting to greyscale");

	dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
	dimGrid = dim3(6, 1, 1);
	wbTime_start(Compute, "Calculating histogram");
	calcHistogram << <dimGrid, dimBlock >> >(deviceGreyImage, deviceHistogram, imageWidth*imageHeight);
	cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
	wbTime_stop(Compute, "Calculating histogram");

	dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
	dimGrid = dim3(1, 1, 1);
	wbTime_start(Compute, "Calculating CDF");
	calcCDF<<<dimGrid, dimBlock>>>(deviceHistogram, deviceCDF, (float)(1.0/(imageWidth*imageHeight)));
	cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
	wbTime_stop(Compute, "Calculating CDF");

	dimBlock = dim3(HISTOGRAM_LENGTH/2, 1, 1);
	dimGrid = dim3(1, 1, 1);
	wbTime_start(Compute, "Calculating CDF min");
	minimum<<<dimGrid, dimBlock>>>(deviceCDF, deviceCDFMin);
	cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
	wbTime_stop(Compute, "Calculating CDF min");

	dimBlock = dim3(TILE_WIDTH*TILE_WIDTH, 1, 1);
	dimGrid = dim3((imageWidth*imageHeight*imageChannels - 1) / (TILE_WIDTH*TILE_WIDTH) + 1, 1, 1);
	wbTime_start(Compute, "equalize uchar image");
	equalizeImage<<<dimGrid, dimBlock >>>(deviceUCharImage, deviceCDF, deviceCDFMin, imageWidth*imageHeight*imageChannels);
	cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
	wbTime_stop(Compute, "equalize uchar image");

	dimBlock = dim3(TILE_WIDTH*TILE_WIDTH, 1, 1);
	dimGrid = dim3((imageWidth*imageHeight*imageChannels - 1) / (TILE_WIDTH*TILE_WIDTH) + 1, 1, 1);
	wbTime_start(Compute, "uchar image to float");
	castToFloat<< <dimGrid, dimBlock >> >(deviceInputImageData, deviceUCharImage, imageWidth*imageHeight*imageChannels);
	cudaDeviceSynchronize();
	wbCheck(cudaGetLastError());
	wbTime_stop(Compute, "uchar image to float");

	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(
		hostOutputImageData,
		deviceInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceGreyImage);
	cudaFree(deviceUCharImage);
	cudaFree(deviceHistogram);
	cudaFree(deviceCDF);
	cudaFree(deviceCDFMin);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
