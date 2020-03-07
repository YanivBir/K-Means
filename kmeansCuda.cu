#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h> 
#include <math.h>
#include "Point.h"
#include "mpi.h"

#define BIG_double (INFINITY)
#define MAX_CUDA_THREADS_BLOCK 512
#define MAX_POINTS_THREAD 100

cudaError_t cuda_setCurrDistErr(cudaError_t cudaStatus, char msg[], Point *points);
cudaError_t cuda_findDiamErr(cudaError_t cudaStatus, char msg[], Point* dev_clusterPoints, double *dev_distancesArr, double *distancesArr);
cudaError_t cuda_calcPointsErr(cudaError_t cudaStatus, char msg[], Point *dev_points, Point *dev_centers, double *dev_distance_output);

//calc distance between 2 points
__device__ double cuda_calcDistancePoints(Point *p1, Point *p2){
	double x, y, z;	
	x = p1->x - p2->x;
	y = p1->y - p2->y;
	z = p1->z - p2->z;
	x *= x;
	y *= y;
	z *= z;
	return sqrt(x + y + z);
}

//set the new posion of points at time
__device__ void cuda_setCurrentPosion(Point *p, double time){
	p->x = (p->x + (time*p->vX));
	p->y = (p->y + (time*p->vY));
	p->z = (p->z + (time*p->vZ));
	p->vX = 0;
	p->vY = 0;
	p->vZ = 0;
}

//updaete the new points posion
__global__ void setCurrentDistance(Point *points, int n, double time){
	int pointId = (blockIdx.x * MAX_CUDA_THREADS_BLOCK) + threadIdx.x;
	if (pointId < n) {
		cuda_setCurrentPosion(&points[pointId], time);
	}
}

//clusterCount is num of points in the cluster
__global__ void findMaxDistance(Point *dev_clusterPoints, int clusterCount, double *dev_distancesArr, int index) {
	int pointId = (blockIdx.x * blockDim.x) + threadIdx.x;
	int pointOffset;
	if (pointId < clusterCount)
	{
		pointOffset = pointId + index;
		for (int i = pointOffset +1; i < pointOffset+1+MAX_POINTS_THREAD; i++)
		{
			double curDistance = cuda_calcDistancePoints(&dev_clusterPoints[pointId], &dev_clusterPoints[i%clusterCount]);
			if (curDistance > dev_distancesArr[pointId])
				dev_distancesArr[pointId] = curDistance;
		}
	}
}

__global__ void pointsDistance(Point *dev_points, int n, Point *dev_centers, int k, double *dev_distance_output) {
	int pointId = (blockIdx.x * MAX_CUDA_THREADS_BLOCK) + threadIdx.x;
	if (pointId < n) {
		for (int j = 0; j < k; j++) { // for each cluster
			dev_distance_output[pointId*k + j] = cuda_calcDistancePoints(&dev_points[pointId], &dev_centers[j]);
		}
	}
}

//calc the current position
cudaError_t cuda_setCurrentPosition(Point *points, int n, double time){
	Point *dev_points = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cuda_setCurrDistErr(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", dev_points);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_points, n * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		cuda_setCurrDistErr(cudaStatus, "cudaMalloc dev_points failed!\n", dev_points);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_points, points, n * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cuda_setCurrDistErr(cudaStatus, "cudaMemcpyHostToDevice dev_points failed!\n", dev_points);
		return cudaStatus;
	}

	int cudaBlocks = n / MAX_CUDA_THREADS_BLOCK;
	if (n % MAX_CUDA_THREADS_BLOCK != 0)
		cudaBlocks++;
	setCurrentDistance << <cudaBlocks, MAX_CUDA_THREADS_BLOCK >> >(dev_points, n, time);

	cudaStatus = cudaMemcpy(points, dev_points, n * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cuda_setCurrDistErr(cudaStatus, "cudaMemcpyHostToDevice dev_points failed!\n", dev_points);
		return cudaStatus;
	}

	cudaFree(dev_points);
	return cudaStatus;
}

//calc the points distance
cudaError_t cuda_calcPointsDistance(Point *points, int n, Point *centers, int k, double *distance_output){
	Point *dev_points = 0;
	Point *dev_centers = 0;
	double *dev_distance_output = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cuda_calcPointsErr(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", dev_points, dev_centers, dev_distance_output);
		return cudaStatus;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_points, n * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		cuda_calcPointsErr(cudaStatus, "cudaMalloc dev_points failed!\n", dev_points, dev_centers, dev_distance_output);
		return cudaStatus;
	}
	
	cudaStatus = cudaMemcpy(dev_points, points, n * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cuda_calcPointsErr(cudaStatus, "cudaMemcpyHostToDevice dev_points failed!\n", dev_points, dev_centers, dev_distance_output);
		return cudaStatus;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_centers, k * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		cuda_calcPointsErr(cudaStatus, "cudaMalloc dec_centers failed!\n", dev_points, dev_centers, dev_distance_output);
		return cudaStatus;
	}
	
	cudaStatus = cudaMemcpy(dev_centers, centers, k * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cuda_calcPointsErr(cudaStatus, "cudaMemcpyHostToDevice dec_centers failed!\n", dev_points, dev_centers, dev_distance_output);
		return cudaStatus;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_distance_output, n*k * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		cuda_calcPointsErr(cudaStatus, "cudaMalloc dev_distance_output failed!\n", dev_points, dev_centers, dev_distance_output);
		return cudaStatus;
	}
	
	int cudaBlocks = n / MAX_CUDA_THREADS_BLOCK;
	if (n % MAX_CUDA_THREADS_BLOCK != 0)
		cudaBlocks++;

	pointsDistance << <cudaBlocks, MAX_CUDA_THREADS_BLOCK >> >(dev_points, n, dev_centers, k, dev_distance_output);
	cudaStatus = cudaMemcpy(distance_output, dev_distance_output, n*k * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cuda_calcPointsErr(cudaStatus, "cudaMemcpyHostToDevice dev_distance_output failed!\n", dev_points, dev_centers, dev_distance_output);
		return cudaStatus;
	}

	cudaFree(dev_points);
	cudaFree(dev_centers);
	cudaFree(dev_distance_output);
	return cudaStatus;
}

//clusterCount is the num of the points that ralated to the cluster
cudaError_t cuda_findDiameter(Point *clusterPoints, int clusterCount, double *diamter){
	Point *dev_clusterPoints = 0;
	double *dev_distancesArr = 0;
	double *distancesArr = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cuda_findDiamErr(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", dev_clusterPoints, dev_distancesArr, distancesArr);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_clusterPoints, clusterCount * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		cuda_findDiamErr(cudaStatus, "cudaMalloc dev_clusterPoints failed!\n", dev_clusterPoints, dev_distancesArr, distancesArr);
		return cudaStatus;
	}
	cudaStatus = cudaMemcpy(dev_clusterPoints, clusterPoints, clusterCount * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cuda_findDiamErr(cudaStatus, "cudaMemcpyHostToDevice dev_clusterPoints failed!\n", dev_clusterPoints, dev_distancesArr, distancesArr);
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&dev_distancesArr, clusterCount * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		cuda_findDiamErr(cudaStatus, "cudaMalloc dev_distancesArr failed!\n", dev_clusterPoints, dev_distancesArr, distancesArr);
		return cudaStatus;
	}

	distancesArr = (double *)malloc(sizeof(double)*clusterCount);
	if (distancesArr == NULL) {
		cuda_findDiamErr(cudaStatus, "distancesArr malloc  failed!\n", dev_clusterPoints, dev_distancesArr, distancesArr);
		return cudaStatus;
	}
	for (int index = 0; index < clusterCount; index++)
	{
		distancesArr[index] = 0;
	}

	cudaStatus = cudaMemcpy(dev_distancesArr, distancesArr, clusterCount * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cuda_findDiamErr(cudaStatus, "cudaMemcpyHostToDevice dev_distancesArr failed!\n", dev_clusterPoints, dev_distancesArr, distancesArr);
		return cudaStatus;
	}

	int cudaBlocks = clusterCount / MAX_CUDA_THREADS_BLOCK;
	if (clusterCount % MAX_CUDA_THREADS_BLOCK != 0)
		cudaBlocks++;

	for (int i = 0; i < clusterCount; i += MAX_POINTS_THREAD)
		findMaxDistance << <cudaBlocks, MAX_CUDA_THREADS_BLOCK >> >(dev_clusterPoints, clusterCount, dev_distancesArr, i);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		cuda_findDiamErr(cudaStatus, "findMaxDistance launch failed!\n", dev_clusterPoints, dev_distancesArr, distancesArr);
		return cudaStatus;
	}
	cudaFree(dev_clusterPoints);

	// Waits until all threads done
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		cuda_findDiamErr(cudaStatus, "cudaDeviceSynchronize failed!\n", dev_clusterPoints, dev_distancesArr, distancesArr);
		return cudaStatus;
	}

	//Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(distancesArr, dev_distancesArr, clusterCount * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cuda_findDiamErr(cudaStatus, "cudaMemcpy  failed!\n", dev_clusterPoints, dev_distancesArr, distancesArr);
		return cudaStatus;
	}
	cudaFree(dev_distancesArr);

	double maxDiamter = distancesArr[0];
	for (int i = 1; i < clusterCount; i++)
	{
		if (distancesArr[i] > maxDiamter)
			maxDiamter = distancesArr[i];
	}
	*diamter = maxDiamter;
	free(distancesArr);

	return cudaStatus;
}

cudaError_t cuda_setCurrDistErr(cudaError_t cudaStatus, char msg[], Point *points){
	cudaFree(points);
	fprintf(stderr, msg);
	return cudaStatus;
}

cudaError_t cuda_findDiamErr(cudaError_t cudaStatus, char msg[], Point* dev_clusterPoints, double *dev_distancesArr, double *distancesArr){
	cudaFree(dev_clusterPoints);
	cudaFree(dev_distancesArr);
	free(distancesArr);
	fprintf(stderr, msg);
	return cudaStatus;
}

cudaError_t cuda_calcPointsErr(cudaError_t cudaStatus, char msg[], Point *dev_points, Point *dev_centers, double *dev_distance_output){
	cudaFree(dev_points);
	cudaFree(dev_centers);
	cudaFree(dev_distance_output);
	fprintf(stderr, msg);
	return cudaStatus;
}

int finalizeCuda(){
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}
