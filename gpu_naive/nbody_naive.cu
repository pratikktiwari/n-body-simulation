#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <utility>
#include <cuda_runtime.h>

#ifndef INITIAL_SEED
#define INITIAL_SEED 42
#endif

// Number of bodies in the simulation
const int NUM_BODIES = 10000;

// Simulation steps
const int NUM_STEPS = 1000;

// Time step for simulation
const float TIME_STEP = 0.005f;

// Save every n steps (0 means do not save)
const int SAVE_EVERY = 2;

// Universal gravitational constant
const float GRAVITATIONAL_CONSTANT = 0.001; // 6.67430e-11f;

// Softening factor to prevent singularities in force calculation
const float SOFTENING = 0.01f;

// CUDA block size for kernel launches
const int BLOCK_SIZE = 256;

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Bodies start in a thin ring. The w component stores mass.
void initBodies(float4* position, float3* velocity, int numBodies)
{
    srand(INITIAL_SEED);
    for (int i = 0; i < numBodies; i++)
    {
        float radius = 1.0f + (rand() / (float)RAND_MAX) * 3.0f;
        float angle = (rand() / (float)RAND_MAX) * 2.0f * M_PI;

        position[i].x = radius * cosf(angle);
        position[i].y = radius * sinf(angle);
        position[i].z = ((rand() / (float)RAND_MAX) - 0.5f) * 0.5f;
        position[i].w = 1.0f + (rand() / (float)RAND_MAX) * 5.0f;

        float speed = 0.5f * sqrtf(1.0f / radius);
        velocity[i].x = -speed * sinf(angle);
        velocity[i].y = speed * cosf(angle);
        velocity[i].z = 0.0f;
    }
}

// Add body2's pull to body1's acceleration.
__device__ float3 addForce(
    float4 body1, float4 body2, float3 acceleration, float gravitationalConstant, float softening)
{
    float deltaX = body2.x - body1.x;
    float deltaY = body2.y - body1.y;
    float deltaZ = body2.z - body1.z;

    // Softening keeps the denominator above zero when positions match.
    float distanceSquared = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + softening;
    float inverseDistance = rsqrtf(distanceSquared);
    // The displacement below needs a scale of 1 / distance^3.
    float accelerationScale =
        gravitationalConstant * body2.w * inverseDistance * inverseDistance * inverseDistance;

    acceleration.x += deltaX * accelerationScale;
    acceleration.y += deltaY * accelerationScale;
    acceleration.z += deltaZ * accelerationScale;
    return acceleration;
}

// Each thread handles one body and loops over all source bodies.
// Each thread reads source bodies directly from global memory.
__global__ void updateBodies(const float4* position, float3* velocity, float4* newPosition, int numBodies,
    float timeStep, float gravitationalConstant, float softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Skip extra threads in the last block.
    if (i >= numBodies)
    {
        return;
    }

    float4 body1 = position[i];
    float3 acceleration = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < numBodies; j++)
    {
        acceleration = addForce(body1, position[j], acceleration, gravitationalConstant, softening);
    }

    // Update velocity first, then use it to move the body.
    float3 updatedVelocity = velocity[i];
    updatedVelocity.x += acceleration.x * timeStep;
    updatedVelocity.y += acceleration.y * timeStep;
    updatedVelocity.z += acceleration.z * timeStep;
    velocity[i] = updatedVelocity;

    newPosition[i] = make_float4(body1.x + updatedVelocity.x * timeStep,
        body1.y + updatedVelocity.y * timeStep, body1.z + updatedVelocity.z * timeStep, body1.w);
}

void saveFrame(FILE* file, int step, const float4* position, int numBodies)
{
    for (int i = 0; i < numBodies; i++)
    {
        fprintf(
            file, "%d,%d,%f,%f,%f,%f\n", step, i, position[i].x, position[i].y, position[i].z, position[i].w);
    }
}

void printReport(const char* version, int numBodies, int numSteps, double setupMs, double computeMs,
    double gpuMs, double saveMs, double totalMs, double cpuMs, float timeStep, int saveEvery,
    float gravitationalConstant, float softening)
{
    // About 20 floating point operations per pair of bodies.
    double pairs = (double)numBodies * numBodies * numSteps;
    printf("version              : %s\n", version);
    printf("random seed          : %u\n", (unsigned int)INITIAL_SEED);
    printf("bodies               : %d\n", numBodies);
    printf("steps                : %d\n", numSteps);
    printf("time step            : %.9g\n", timeStep);
    printf("save every           : %d\n", saveEvery);
    printf("gravitational constant: %.9g\n", gravitationalConstant);
    printf("softening            : %.9g\n", softening);
    printf("block size           : %d\n", BLOCK_SIZE);
    printf("setup time ms        : %.3f\n", setupMs);
    printf("compute time ms      : %.3f\n", computeMs);
    printf("gpu time ms          : %.3f\n", gpuMs);
    printf("saving time ms       : %.3f\n", saveMs);
    printf("total time ms        : %.3f\n", totalMs);
    printf("cpu time ms          : %.3f\n", cpuMs);
    printf("ms per step          : %.4f\n", computeMs / numSteps);
    printf("billion pairs per sec: %.3f\n", pairs / (computeMs * 1e6));
    printf("gflops               : %.2f\n", 20.0 * pairs / (computeMs * 1e6));
}

int main(int argc, char** argv)
{
    // ./nbody_naive.out [bodies] [steps] [timeStep] [saveEvery] [gravitationalConstant] [softening]
    int numBodies = (argc > 1) ? atoi(argv[1]) : NUM_BODIES;
    int numSteps = (argc > 2) ? atoi(argv[2]) : NUM_STEPS;
    float timeStep = (argc > 3) ? atof(argv[3]) : TIME_STEP;
    int saveEvery = (argc > 4) ? atoi(argv[4]) : SAVE_EVERY; // 0 means no frames are saved.
    float gravitationalConstant = (argc > 5) ? atof(argv[5]) : GRAVITATIONAL_CONSTANT;
    float softening = (argc > 6) ? atof(argv[6]) : SOFTENING;

    double programStart = nowMs();
    clock_t cpuStart = clock();

    float4* position = new float4[numBodies];
    float3* velocity = new float3[numBodies];
    initBodies(position, velocity, numBodies);

    // Setup time includes starting the GPU on the first CUDA call.
    float4 *gpuPosition, *gpuNewPosition;
    float3* gpuVelocity;
    cudaMalloc(&gpuPosition, numBodies * sizeof(float4));
    cudaMalloc(&gpuNewPosition, numBodies * sizeof(float4));
    cudaMalloc(&gpuVelocity, numBodies * sizeof(float3));
    cudaMemcpy(gpuPosition, position, numBodies * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuVelocity, velocity, numBodies * sizeof(float3), cudaMemcpyHostToDevice);

    int numBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    FILE* framesFile = fopen(POSITIONS_FILE, "w");
    fprintf(framesFile, "step,id,x,y,z,mass\n");

    double setupMs = nowMs() - programStart;
    double gpuMs = 0.0;
    double saveMs = 0.0;
    if (saveEvery > 0)
    {
        double saveStart = nowMs();
        saveFrame(framesFile, 0, position, numBodies);
        saveMs += nowMs() - saveStart;
    }

    for (int step = 0; step < numSteps; step++)
    {
        // Time the kernel only, without frame copies or file writes.
        cudaEventRecord(start);
        updateBodies<<<numBlocks, BLOCK_SIZE>>>(
            gpuPosition, gpuVelocity, gpuNewPosition, numBodies, timeStep, gravitationalConstant, softening);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float stepMs = 0.0f;
        cudaEventElapsedTime(&stepMs, start, stop);
        gpuMs += stepMs;
        // Use the new positions as input for the next step.
        std::swap(gpuPosition, gpuNewPosition);

        // Copy positions back to the CPU before saving a frame.
        if (saveEvery > 0 && (step + 1) % saveEvery == 0)
        {
            double saveStart = nowMs();
            cudaMemcpy(position, gpuPosition, numBodies * sizeof(float4), cudaMemcpyDeviceToHost);
            saveFrame(framesFile, step + 1, position, numBodies);
            saveMs += nowMs() - saveStart;
        }
    }
    fclose(framesFile);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(position, gpuPosition, numBodies * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocity, gpuVelocity, numBodies * sizeof(float3), cudaMemcpyDeviceToHost);

    FILE* finalFile = fopen(FINAL_STATE_FILE, "w");
    fprintf(finalFile, "id,x,y,z,vx,vy,vz,mass\n");
    for (int i = 0; i < numBodies; i++)
    {
        fprintf(finalFile, "%d,%f,%f,%f,%f,%f,%f,%f\n", i, position[i].x, position[i].y, position[i].z,
            velocity[i].x, velocity[i].y, velocity[i].z, position[i].w);
    }
    fclose(finalFile);

    cudaFree(gpuPosition);
    cudaFree(gpuNewPosition);
    cudaFree(gpuVelocity);
    delete[] position;
    delete[] velocity;

    double totalMs = nowMs() - programStart;
    double cpuMs = 1000.0 * (clock() - cpuStart) / CLOCKS_PER_SEC;
    printReport("v2 naive GPU", numBodies, numSteps, setupMs, gpuMs, gpuMs, saveMs, totalMs, cpuMs, timeStep,
        saveEvery, gravitationalConstant, softening);
    return 0;
}
