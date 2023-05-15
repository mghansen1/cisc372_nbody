#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void ParallelCompCuda(vector3 *hPos, vector3 *accels, double *mass) {
    int i = (blockDim.y * blockIdx.y) + threadIdx.y; //row
    int j = (blockDim.x * blockIdx.x) + threadIdx.x; //col
    int idx = (NUMENTITIES * i) + j; 

    if (idx < NUMENTITIES * NUMENTITIES) {
        if (i == j) {
            FILL_VECTOR(accels[idx],0,0,0);
        } else {
            double x = (hPos[i][0] - hPos[j][0]);
            double y = (hPos[i][1] - hPos[j][1]);
            double z = (hPos[i][2] - hPos[j][2]);
            double mag_sq = x*x + y*y + z*z;
            double mag = sqrt( mag_sq );
            double accel = -1 *GRAV_CONSTANT * mass[j] / mag_sq;
            FILL_VECTOR(accels[idx], accel * x / mag, accel * y / mag, accel * z/ mag);

        }
    }
}

__global__ void ParallelSumCuda(vector3 *accels, vector3 *accel_sum, vector3 *hPos, vector3 *hVel){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NUMENTITIES){
		FILL_VECTOR(accel_sum[i], 0, 0, 0);
		for (int j = 0; j < NUMENTITIES; j++){
			for (int k = 0; k < 3; k++){
				accel_sum[i][k] += accels[(i * NUMENTITIES) + j][k];
			}
		}
		for (int k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[i][k] * INTERVAL;
			hPos[i][k] = hVel[i][k] * INTERVAL;
		}

	}
}

void compute() {
    vector3 *dev_hPos, *dev_hVel, *dev_acc, *dev_sum;
    double *dev_mass;
	
    int blocksD = ceilf( NUMENTITIES / 16.0f  );
    int threadsD = ceilf( NUMENTITIES / (float)blocksD );
    dim3 gridDim(blocksD, blocksD, 1);
    dim3 blockDim(threadsD, threadsD, 1);

    cudaMalloc( &dev_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc( &dev_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc( &dev_acc, sizeof(vector3) * NUMENTITIES);
    cudaMalloc( &dev_sum, sizeof(vector3) * NUMENTITIES);
    cudaMalloc( &dev_mass, sizeof(double) * NUMENTITIES);

    cudaMemcpy(dev_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

    ParallelCompCuda<<<gridDim, blockDim>>>(dev_hPos, dev_acc, dev_mass);
    cudaDeviceSynchronize();

    ParallelSumCuda<<<gridDim.x, blockDim.x>>>(dev_acc, dev_sum, dev_hPos, dev_hVel);
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, dev_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dev_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

    cudaFree(dev_hPos);
    cudaFree(dev_hVel);
    cudaFree(dev_mass);
    cudaFree(dev_acc);

}
