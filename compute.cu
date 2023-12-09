#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define BLOCK_SIZE 16

__global__ void computeAccelsAndSum(vector3* accels, vector3* hPos, double* mass, vector3* hVel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NUMENTITIES && j < NUMENTITIES) {
        vector3 distance;
        if (i != j) {
            for (int k = 0; k < 3; k++) {
                distance[k] = hPos[i][k] - hPos[j][k];
            }
            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
            accels[i * NUMENTITIES + j][0] = accelmag * distance[0] / magnitude;
            accels[i * NUMENTITIES + j][1] = accelmag * distance[1] / magnitude;
            accels[i * NUMENTITIES + j][2] = accelmag * distance[2] / magnitude;
        } else {
            FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
        }

        __syncthreads();  // Synchronize threads within the block

        // Sum up the rows of our matrix to get effect on each entity
        vector3 accel_sum = {0, 0, 0};
        for (int k = 0; k < NUMENTITIES; k++) {
            accel_sum[0] += accels[k * NUMENTITIES + j][0];
            accel_sum[1] += accels[k * NUMENTITIES + j][1];
            accel_sum[2] += accels[k * NUMENTITIES + j][2];
        }

        // Compute the new velocity based on the acceleration and time interval
        // Compute the new position based on the velocity and time interval
        for (int k = 0; k < 3; k++) {
            hVel[i][k] += accel_sum[k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
        }
    }
}

void compute() {
    // Allocate GPU memory for accels, hPos, hVel, and mass
    vector3* dAccels;
    cudaMalloc((void**)&dAccels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    vector3* dPos;
    cudaMalloc((void**)&dPos, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(dPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    vector3* dVel;
    cudaMalloc((void**)&dVel, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(dVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    double* dMass;
    cudaMalloc((void**)&dMass, sizeof(double) * NUMENTITIES);
    cudaMemcpy(dMass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blocksPerGrid((NUMENTITIES + BLOCK_SIZE - 1) / BLOCK_SIZE, (NUMENTITIES + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Launch the kernel to compute accelerations and update positions
    computeAccelsAndSum<<<blocksPerGrid, threadsPerBlock>>>(dAccels, dPos, dMass, dVel);

    // Synchronize to ensure the kernel is complete
    cudaDeviceSynchronize();

    // Copy results from device to host if needed

    // Free GPU memory
    cudaFree(dAccels);
    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dMass);
}
