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

// 0 tries the available slice counts.
const int NUM_SLICES = 0;

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

__device__ float3 addForce(
    float4 body1, float4 body2, float3 acceleration, float gravitationalConstant, float softening)
{
    float deltaX = body2.x - body1.x;
    float deltaY = body2.y - body1.y;
    float deltaZ = body2.z - body1.z;

    float distanceSquared = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + softening;
    float inverseDistance = rsqrtf(distanceSquared);
    float accelerationScale =
        gravitationalConstant * body2.w * inverseDistance * inverseDistance * inverseDistance;

    acceleration.x += deltaX * accelerationScale;
    acceleration.y += deltaY * accelerationScale;
    acceleration.z += deltaZ * accelerationScale;
    return acceleration;
}

// Split source bodies into slices to give each target body more threads.
// Each warp handles one slice for the same 32 bodies.
__global__ void updateBodies(const float4* position, float3* velocity, float4* newPosition, int numBodies,
    float timeStep, float gravitationalConstant, float softening)
{
    extern __shared__ float3 partialAccelerations[];

    int numSlices = blockDim.x / 32;
    int lane = threadIdx.x % 32;
    int slice = threadIdx.x / 32;
    int i = blockIdx.x * 32 + lane;

    float4 body1 = (i < numBodies) ? position[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 acceleration = make_float3(0.0f, 0.0f, 0.0f);

    // Threads in a warp read the same source body through the cache.
    int sliceSize = (numBodies + numSlices - 1) / numSlices;
    int first = slice * sliceSize;
    int last = min(first + sliceSize, numBodies);
    for (int j = first; j < last; j++)
    {
        acceleration = addForce(body1, position[j], acceleration, gravitationalConstant, softening);
    }

    partialAccelerations[threadIdx.x] = acceleration;
    __syncthreads();

    // The first warp adds the partial accelerations, then moves the bodies.
    if (slice == 0 && i < numBodies)
    {
        for (int s = 1; s < numSlices; s++)
        {
            acceleration.x += partialAccelerations[s * 32 + lane].x;
            acceleration.y += partialAccelerations[s * 32 + lane].y;
            acceleration.z += partialAccelerations[s * 32 + lane].z;
        }

        // Update velocity first, then use it to move the body.
        float3 newVelocity = velocity[i];
        newVelocity.x += acceleration.x * timeStep;
        newVelocity.y += acceleration.y * timeStep;
        newVelocity.z += acceleration.z * timeStep;
        velocity[i] = newVelocity;

        newPosition[i] = make_float4(body1.x + newVelocity.x * timeStep, body1.y + newVelocity.y * timeStep,
            body1.z + newVelocity.z * timeStep, body1.w);
    }
}

void runStep(int numSlices, const float4* position, float3* velocity, float4* newPosition, int numBodies,
    float timeStep, float gravitationalConstant, float softening)
{
    int numBlocks = (numBodies + 31) / 32;
    int threads = 32 * numSlices;
    updateBodies<<<numBlocks, threads, threads * sizeof(float3)>>>(
        position, velocity, newPosition, numBodies, timeStep, gravitationalConstant, softening);
}

// Try each slice count on copied bodies and keep the fastest.
int pickSlices(const float4* gpuPosition, const float3* gpuVelocity, int numBodies, float timeStep,
    float gravitationalConstant, float softening)
{
    int options[] = {1, 2, 4, 8, 16, 32};
    int trialSteps = 10;

    float4 *testPosition, *testNewPosition;
    float3* testVelocity;
    cudaMalloc(&testPosition, numBodies * sizeof(float4));
    cudaMalloc(&testNewPosition, numBodies * sizeof(float4));
    cudaMalloc(&testVelocity, numBodies * sizeof(float3));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int best = 1;
    float bestMs = 1e30f;
    printf("trying slices per body:\n");
    for (int numSlices : options)
    {
        cudaMemcpy(testPosition, gpuPosition, numBodies * sizeof(float4), cudaMemcpyDeviceToDevice);
        cudaMemcpy(testVelocity, gpuVelocity, numBodies * sizeof(float3), cudaMemcpyDeviceToDevice);
        runStep(numSlices, testPosition, testVelocity, testNewPosition, numBodies, timeStep,
            gravitationalConstant,
            softening); // Warm-up step.

        cudaEventRecord(start);
        for (int t = 0; t < trialSteps; t++)
        {
            runStep(numSlices, testPosition, testVelocity, testNewPosition, numBodies, timeStep,
                gravitationalConstant, softening);
            std::swap(testPosition, testNewPosition);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= trialSteps;
        printf("  %2d slices: %.4f ms per step\n", numSlices, ms);
        if (ms < bestMs)
        {
            bestMs = ms;
            best = numSlices;
        }
    }
    printf("using %d slices\n", best);

    cudaFree(testPosition);
    cudaFree(testNewPosition);
    cudaFree(testVelocity);
    return best;
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
    float gravitationalConstant, float softening, int numSlices)
{
    // About 20 operations per pair when rsqrtf counts as one.
    double pairs = (double)numBodies * numBodies * numSteps;
    printf("version              : %s\n", version);
    printf("random seed          : %u\n", (unsigned int)INITIAL_SEED);
    printf("bodies               : %d\n", numBodies);
    printf("steps                : %d\n", numSteps);
    printf("time step            : %.9g\n", timeStep);
    printf("save every           : %d\n", saveEvery);
    printf("gravitational constant: %.9g\n", gravitationalConstant);
    printf("softening            : %.9g\n", softening);
    printf("slices per body      : %d\n", numSlices);
    printf("block size           : %d\n", 32 * numSlices);
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
    // ./nbody_opt.out [bodies] [steps] [timeStep] [saveEvery] [gravitationalConstant] [softening] [slices]
    // 0 tries the available slice counts and picks the fastest.
    int numBodies = (argc > 1) ? atoi(argv[1]) : NUM_BODIES;
    int numSteps = (argc > 2) ? atoi(argv[2]) : NUM_STEPS;
    float timeStep = (argc > 3) ? atof(argv[3]) : TIME_STEP;
    int saveEvery = (argc > 4) ? atoi(argv[4]) : SAVE_EVERY;
    float gravitationalConstant = (argc > 5) ? atof(argv[5]) : GRAVITATIONAL_CONSTANT;
    float softening = (argc > 6) ? atof(argv[6]) : SOFTENING;
    int numSlices = (argc > 7) ? atoi(argv[7]) : NUM_SLICES;

    double programStart = nowMs();
    clock_t cpuStart = clock();

    float4* position = new float4[numBodies];
    float3* velocity = new float3[numBodies];
    initBodies(position, velocity, numBodies);

    float4 *gpuPosition, *gpuNewPosition;
    float3* gpuVelocity;
    cudaMalloc(&gpuPosition, numBodies * sizeof(float4));
    cudaMalloc(&gpuNewPosition, numBodies * sizeof(float4));
    cudaMalloc(&gpuVelocity, numBodies * sizeof(float3));
    cudaMemcpy(gpuPosition, position, numBodies * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuVelocity, velocity, numBodies * sizeof(float3), cudaMemcpyHostToDevice);

    // Saved frames stay on the GPU until the run ends.
    int numFrames = (saveEvery > 0) ? numSteps / saveEvery + 1 : 0;
    float4* gpuFrames = nullptr;
    if (numFrames > 0)
    {
        cudaMalloc(&gpuFrames, (size_t)numFrames * numBodies * sizeof(float4));
    }

    if (numSlices <= 0)
    {
        numSlices =
            pickSlices(gpuPosition, gpuVelocity, numBodies, timeStep, gravitationalConstant, softening);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double setupMs = nowMs() - programStart;

    // The CPU queues steps and frame copies without waiting between them.
    cudaEventRecord(start);
    int frame = 0;
    if (saveEvery > 0)
    {
        cudaMemcpyAsync(gpuFrames, gpuPosition, numBodies * sizeof(float4), cudaMemcpyDeviceToDevice);
        frame++;
    }
    for (int step = 0; step < numSteps; step++)
    {
        runStep(numSlices, gpuPosition, gpuVelocity, gpuNewPosition, numBodies, timeStep,
            gravitationalConstant, softening);
        std::swap(gpuPosition, gpuNewPosition);

        if (saveEvery > 0 && (step + 1) % saveEvery == 0)
        {
            cudaMemcpyAsync(gpuFrames + (size_t)frame * numBodies, gpuPosition, numBodies * sizeof(float4),
                cudaMemcpyDeviceToDevice);
            frame++;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuMs = 0.0f;
    cudaEventElapsedTime(&gpuMs, start, stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy all frames back together, then write them to the file.
    double saveStart = nowMs();
    FILE* framesFile = fopen(POSITIONS_FILE, "w");
    if (framesFile == nullptr)
    {
        perror(POSITIONS_FILE);
        return 1;
    }
    fprintf(framesFile, "step,id,x,y,z,mass\n");
    if (frame > 0)
    {
        float4* frames = new float4[(size_t)frame * numBodies];
        cudaMemcpy(frames, gpuFrames, (size_t)frame * numBodies * sizeof(float4), cudaMemcpyDeviceToHost);
        for (int f = 0; f < frame; f++)
        {
            saveFrame(framesFile, f * saveEvery, frames + (size_t)f * numBodies, numBodies);
        }
        delete[] frames;
    }
    fclose(framesFile);
    double saveMs = nowMs() - saveStart;

    cudaMemcpy(position, gpuPosition, numBodies * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocity, gpuVelocity, numBodies * sizeof(float3), cudaMemcpyDeviceToHost);

    FILE* finalFile = fopen(FINAL_STATE_FILE, "w");
    if (finalFile == nullptr)
    {
        perror(FINAL_STATE_FILE);
        return 1;
    }
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

    if (gpuFrames)
    {
        cudaFree(gpuFrames);
    }

    delete[] position;
    delete[] velocity;

    double totalMs = nowMs() - programStart;
    double cpuMs = 1000.0 * (clock() - cpuStart) / CLOCKS_PER_SEC;
    printReport("v4 optimised GPU", numBodies, numSteps, setupMs, gpuMs, gpuMs, saveMs, totalMs, cpuMs,
        timeStep, saveEvery, gravitationalConstant, softening, numSlices);
    return 0;
}
