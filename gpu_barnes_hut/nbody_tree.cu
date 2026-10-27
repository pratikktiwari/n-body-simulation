#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#ifndef LEAF
#define LEAF 16 // bodies per leaf
#endif
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

// Ring scene and tree settings.
const int SCENARIO = 2;
const float OPENING_ANGLE = 0.6f;
const char* const MODE = "tree";
const int RESORT_EVERY = 4;
const int MAX_FRAME_BODIES = 32768;
const int CHECK_BODIES = 1024;

#define PI_F 3.14159265358979323846f
#define EMPTY_BOX 1.0e30f

// One thread per item.
#ifndef LAUNCH
#define LAUNCH(kernel, count, ...)                                                                           \
    kernel<<<((count) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(__VA_ARGS__)
#endif
#define THREAD_INDEX ((int)(blockIdx.x * blockDim.x + threadIdx.x))

#define CUDA_CHECK(call)                                                                                     \
    do                                                                                                       \
    {                                                                                                        \
        cudaError_t err__ = (call);                                                                          \
        if (err__ != cudaSuccess)                                                                            \
        {                                                                                                    \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err__), __FILE__, __LINE__);      \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)

// Mass is multiplied by G once on the CPU, so w holds G * mass.
__device__ __forceinline__ float3 addForce(float4 body1, float4 body2, float3 acceleration, float softening)
{
    float deltaX = body2.x - body1.x;
    float deltaY = body2.y - body1.y;
    float deltaZ = body2.z - body1.z;
    float inverseDistance = rsqrtf(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + softening);
    float accelerationScale = body2.w * inverseDistance * inverseDistance * inverseDistance;
    acceleration.x += deltaX * accelerationScale;
    acceleration.y += deltaY * accelerationScale;
    acceleration.z += deltaZ * accelerationScale;
    return acceleration;
}

// Spread the lowest 21 bits, leaving two zero bits between them.
__device__ __forceinline__ uint64_t spreadBits(uint64_t v)
{
    v &= 0x1fffffULL;
    v = (v | (v << 32)) & 0x1f00000000ffffULL;
    v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
    v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
    v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
    v = (v | (v << 2)) & 0x1249249249249249ULL;
    return v;
}

// Morton keys use the previous tree's bounding box.
// Bodies outside it are clamped only when making the keys.
__global__ void mortonKernel(const float4* position, const float4* nodeLowerBounds,
    const float4* nodeUpperBounds, uint64_t* keys, uint32_t* permutation, int numBodies)
{
    int i = THREAD_INDEX;
    if (i >= numBodies)
    {
        return;
    }
    float4 lo = nodeLowerBounds[1], hi = nodeUpperBounds[1];
    float extent = fmaxf(fmaxf(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z);
    float scale = 2097152.0f / fmaxf(extent, 1.0e-20f);
    float4 p = position[i];
    uint64_t qx = (uint64_t)fminf(fmaxf((p.x - lo.x) * scale, 0.0f), 2097151.0f);
    uint64_t qy = (uint64_t)fminf(fmaxf((p.y - lo.y) * scale, 0.0f), 2097151.0f);
    uint64_t qz = (uint64_t)fminf(fmaxf((p.z - lo.z) * scale, 0.0f), 2097151.0f);
    keys[i] = spreadBits(qx) | (spreadBits(qy) << 1) | (spreadBits(qz) << 2);
    permutation[i] = (uint32_t)i;
}

// Keep the original IDs so saved frames stay in body order after sorting.
__global__ void gatherKernel(const float4* positionIn, const float4* velocityIn, const uint32_t* idIn,
    const uint32_t* permutation, float4* positionOut, float4* velocityOut, uint32_t* idOut, int numBodies)
{
    int i = THREAD_INDEX;
    if (i >= numBodies)
    {
        return;
    }
    uint32_t src = permutation[i];
    positionOut[i] = positionIn[src];
    velocityOut[i] = velocityIn[src];
    idOut[i] = idIn[src];
}

// Each node stores its mass, centre of mass and squared opening distance.
// The opening distance is size / openingAngle + the offset from the box centre.
// For openingAngle <= 1, a body cannot accept a node that contains it.
__device__ __forceinline__ void finishNode(int k, float mass, float sx, float sy, float sz, float4 lo,
    float4 hi, float openingAngle, float4* nodeCenterOfMass, float* nodeOpeningDistanceSquared,
    float4* nodeLowerBounds, float4* nodeUpperBounds)
{
    nodeLowerBounds[k] = lo;
    nodeUpperBounds[k] = hi;
    if (mass <= 0.0f)
    { // An empty node adds no force.
        nodeCenterOfMass[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        nodeOpeningDistanceSquared[k] = -1.0f;
        return;
    }
    float inv = 1.0f / mass;
    float cx = sx * inv, cy = sy * inv, cz = sz * inv;
    nodeCenterOfMass[k] = make_float4(cx, cy, cz, mass);
    float size = fmaxf(fmaxf(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z);
    float ox = cx - 0.5f * (lo.x + hi.x);
    float oy = cy - 0.5f * (lo.y + hi.y);
    float oz = cz - 0.5f * (lo.z + hi.z);
    float offset = sqrtf(ox * ox + oy * oy + oz * oz);
    float rc = (openingAngle > 0.0f) ? size / openingAngle + offset : 1.0e15f; // An opening angle of 0 opens every nonempty node.
    nodeOpeningDistanceSquared[k] = rc * rc;
}

// Each leaf holds up to LEAF bodies from the sorted array.
__global__ void leafKernel(const float4* position, float4* nodeCenterOfMass,
    float* nodeOpeningDistanceSquared, float4* nodeLowerBounds, float4* nodeUpperBounds, int numBodies,
    int numLeaves, float openingAngle)
{
    int l = THREAD_INDEX;
    if (l >= numLeaves)
    {
        return;
    }
    int first = l * LEAF;
    int last = (first + LEAF < numBodies) ? first + LEAF : numBodies;
    float mass = 0.0f, sx = 0.0f, sy = 0.0f, sz = 0.0f;
    float4 lo = make_float4(EMPTY_BOX, EMPTY_BOX, EMPTY_BOX, 0.0f);
    float4 hi = make_float4(-EMPTY_BOX, -EMPTY_BOX, -EMPTY_BOX, 0.0f);
    for (int j = first; j < last; ++j)
    {
        float4 b = position[j];
        mass += b.w;
        sx += b.w * b.x;
        sy += b.w * b.y;
        sz += b.w * b.z;
        lo.x = fminf(lo.x, b.x);
        lo.y = fminf(lo.y, b.y);
        lo.z = fminf(lo.z, b.z);
        hi.x = fmaxf(hi.x, b.x);
        hi.y = fmaxf(hi.y, b.y);
        hi.z = fmaxf(hi.z, b.z);
    }
    finishNode(numLeaves + l, mass, sx, sy, sz, lo, hi, openingAngle, nodeCenterOfMass,
        nodeOpeningDistanceSquared, nodeLowerBounds, nodeUpperBounds);
}

// Build this level by combining each node's two children.
__global__ void levelKernel(float4* nodeCenterOfMass, float* nodeOpeningDistanceSquared,
    float4* nodeLowerBounds, float4* nodeUpperBounds, int first, float openingAngle)
{
    int k = first + THREAD_INDEX;
    if (k >= 2 * first)
    {
        return;
    }
    float4 a = nodeCenterOfMass[2 * k], b = nodeCenterOfMass[2 * k + 1];
    float4 loA = nodeLowerBounds[2 * k], loB = nodeLowerBounds[2 * k + 1];
    float4 hiA = nodeUpperBounds[2 * k], hiB = nodeUpperBounds[2 * k + 1];
    float4 lo = make_float4(fminf(loA.x, loB.x), fminf(loA.y, loB.y), fminf(loA.z, loB.z), 0.0f);
    float4 hi = make_float4(fmaxf(hiA.x, hiB.x), fmaxf(hiA.y, hiB.y), fmaxf(hiA.z, hiB.z), 0.0f);
    finishNode(k, a.w + b.w, a.w * a.x + b.w * b.x, a.w * a.y + b.w * b.y, a.w * a.z + b.w * b.z, lo, hi,
        openingAngle, nodeCenterOfMass, nodeOpeningDistanceSquared, nodeLowerBounds, nodeUpperBounds);
}

// Move to the right sibling, or the nearest ancestor with a right sibling.
// Removing trailing zero bits from k + 1 finds that sibling without a stack.
// Reaching node 1 again ends the walk.
__device__ __forceinline__ float3 treeAcceleration(float4 body1, const float4* position,
    const float4* nodeCenterOfMass, const float* nodeOpeningDistanceSquared, int numBodies, int numLeaves,
    float softening, int* interactions)
{
    float3 acceleration = make_float3(0.0f, 0.0f, 0.0f);
    int count = 0;
    int k = 1;
    do
    {
        float4 c = nodeCenterOfMass[k];
        float deltaX = c.x - body1.x, deltaY = c.y - body1.y, deltaZ = c.z - body1.z;
        float d2 = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
        bool descend = false;
        if (d2 > nodeOpeningDistanceSquared[k])
        { // Use a distant or empty node as one interaction.
            acceleration = addForce(body1, c, acceleration, softening);
            ++count;
        }
        else if (k >= numLeaves)
        { // Sum nearby leaf bodies directly.
            int first = (k - numLeaves) * LEAF;
            int last = (first + LEAF < numBodies) ? first + LEAF : numBodies;
            for (int j = first; j < last; ++j)
            {
                acceleration = addForce(body1, position[j], acceleration, softening);
            }
            count += last - first;
        }
        else
        {
            descend = true; // Open a nearby internal node.
        }
        if (descend)
        {
            k = 2 * k;
        }
        else
        {
            int n = k + 1;
            k = n >> (__ffs(n) - 1);
        }
    } while (k != 1);
    *interactions = count;
    return acceleration;
}

// Calculate acceleration, update velocity, then move the body.
__global__ void treeStepKernel(const float4* positionIn, float4* velocity, float4* positionOut,
    const float4* nodeCenterOfMass, const float* nodeOpeningDistanceSquared, int numBodies, int numLeaves,
    float timeStep, float softening)
{
    int i = THREAD_INDEX;
    if (i >= numBodies)
    {
        return;
    }
    float4 body = positionIn[i];
    int count;
    float3 acceleration = treeAcceleration(body, positionIn, nodeCenterOfMass, nodeOpeningDistanceSquared,
        numBodies, numLeaves, softening, &count);
    float4 newVelocity = velocity[i];
    newVelocity.x += acceleration.x * timeStep;
    newVelocity.y += acceleration.y * timeStep;
    newVelocity.z += acceleration.z * timeStep;
    velocity[i] = newVelocity;
    body.x += newVelocity.x * timeStep;
    body.y += newVelocity.y * timeStep;
    body.z += newVelocity.z * timeStep;
    positionOut[i] = body;
}

// Direct all-pairs update for mode "direct".
__global__ void directStepKernel(const float4* positionIn, float4* velocity, float4* positionOut,
    int numBodies, float timeStep, float softening)
{
    int i = THREAD_INDEX;
    if (i >= numBodies)
    {
        return;
    }
    float4 body = positionIn[i];
    float3 acceleration = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < numBodies; ++j)
    {
        acceleration = addForce(body, positionIn[j], acceleration, softening);
    }
    float4 newVelocity = velocity[i];
    newVelocity.x += acceleration.x * timeStep;
    newVelocity.y += acceleration.y * timeStep;
    newVelocity.z += acceleration.z * timeStep;
    velocity[i] = newVelocity;
    body.x += newVelocity.x * timeStep;
    body.y += newVelocity.y * timeStep;
    body.z += newVelocity.z * timeStep;
    positionOut[i] = body;
}

// Compare sampled tree forces with direct forces and count the tree interactions.
__global__ void checkKernel(const float4* position, const float4* nodeCenterOfMass,
    const float* nodeOpeningDistanceSquared, int numBodies, int numLeaves, float softening, int M,
    float* relError, int* interactions)
{
    int t = THREAD_INDEX;
    if (t >= M)
    {
        return;
    }
    int i = (int)(((long long)t * numBodies) / M);
    float4 p = position[i];
    int count;
    float3 a = treeAcceleration(
        p, position, nodeCenterOfMass, nodeOpeningDistanceSquared, numBodies, numLeaves, softening, &count);
    float3 e = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < numBodies; ++j)
    {
        e = addForce(p, position[j], e, softening);
    }
    float deltaX = a.x - e.x, deltaY = a.y - e.y, deltaZ = a.z - e.z;
    relError[t] = sqrtf((deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ) /
        fmaxf(e.x * e.x + e.y * e.y + e.z * e.z, 1.0e-30f));
    interactions[t] = count;
}

// Save every stride-th body in original order to keep large frame files smaller.
__global__ void saveFrameKernel(
    const float4* position, const uint32_t* id, float4* frame, int numBodies, int stride)
{
    int i = THREAD_INDEX;
    if (i >= numBodies)
    {
        return;
    }
    uint32_t who = id[i];
    if (who % (uint32_t)stride == 0)
    {
        frame[who / (uint32_t)stride] = position[i];
    }
}

static float frand()
{
    return (float)rand() / (float)RAND_MAX;
}

// Start bodies in a ring with random positions and sideways motion.
void initRing(float4* position, float4* velocity, int numBodies)
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
        velocity[i] = make_float4(-speed * sinf(angle), speed * cosf(angle), 0.0f, 0.0f);
    }
}

// Scenario 0: a uniform sphere starting at rest.
void initSphere(float4* position, float4* velocity, int numBodies)
{
    srand(INITIAL_SEED);
    for (int i = 0; i < numBodies; ++i)
    {
        float r = 1.5f * powf(frand(), 1.0f / 3.0f);
        float openingAngle = frand() * 2.0f * PI_F;
        float phi = acosf(2.0f * frand() - 1.0f);
        position[i].x = r * sinf(phi) * cosf(openingAngle);
        position[i].y = r * sinf(phi) * sinf(openingAngle);
        position[i].z = r * cosf(phi);
        position[i].w = 1.0f + frand() * 9.0f;
        velocity[i].x = velocity[i].y = velocity[i].z = velocity[i].w = 0.0f;
    }
}

// Scenario 1: two rotating disk galaxies on a collision orbit.
// Disk masses scale with body count, so adding bodies mainly adds detail.
void initGalaxies(
    float4* position, float4* velocity, int numBodies, float gravitationalConstant, float softening)
{
    srand(INITIAL_SEED);
    const float coreMass = 1.0f, diskMass = 0.25f;
    const float rMin = 0.08f, rMax = 1.2f, rScale = 0.35f;
    const float sep = 4.5f, impact = 2.0f, orbitFrac = 0.8f;
    const float tilt[2] = {0.26f, 1.05f};

    float totalMass = 2.0f * (coreMass + diskMass);
    float dist = sqrtf(sep * sep + impact * impact);
    float vRel = orbitFrac * sqrtf(2.0f * gravitationalConstant * totalMass / dist);

    int half = numBodies / 2;
    for (int g = 0; g < 2; ++g)
    {
        int begin = g * half;
        int end = (g == 0) ? half : numBodies;
        int n = end - begin;
        float sgn = (g == 0) ? -1.0f : 1.0f;
        float cx = sgn * 0.5f * sep, cy = sgn * 0.5f * impact;
        float cvx = -sgn * 0.5f * vRel;
        float ct = cosf(tilt[g]), st = sinf(tilt[g]);

        for (int i = begin; i < end; ++i)
        {
            float x = 0, y = 0, z = 0, vx = 0, vy = 0, vz = 0, m = coreMass;
            if (i != begin)
            {
                float u = frand();
                float r = rMin - rScale * logf(1.0f - u * (1.0f - expf(-(rMax - rMin) / rScale)));
                float th = frand() * 2.0f * PI_F;
                float mEnc = coreMass + diskMass * u;
                float v = sqrtf(gravitationalConstant * mEnc * r * r / powf(r * r + softening, 1.5f));
                x = r * cosf(th);
                y = r * sinf(th);
                z = 0.04f * (frand() - 0.5f);
                vx = -v * sinf(th);
                vy = v * cosf(th);
                m = diskMass / (float)(n - 1) * (0.5f + frand());
            }
            position[i].x = cx + x;
            position[i].y = cy + y * ct - z * st;
            position[i].z = y * st + z * ct;
            position[i].w = m;
            velocity[i].x = cvx + vx;
            velocity[i].y = vy * ct - vz * st;
            velocity[i].z = vy * st + vz * ct;
            velocity[i].w = 0.0f;
        }
    }
}

static double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Simulation
{
    int numBodies, numLeaves; // Body count and padded leaf count.
    float openingAngle, softening, timeStep;
    float4* position[2]; // Alternate between two buffers.
    float4* velocity[2];
    uint32_t* id[2];
    int positionBuffer, velocityBuffer, idBuffer; // Current buffer indices.
    uint64_t* keys;
    uint32_t* permutation;
    float4 *nodeCenterOfMass, *nodeLowerBounds, *nodeUpperBounds;
    float* nodeOpeningDistanceSquared;

    void sort()
    {
        LAUNCH(mortonKernel, numBodies, position[positionBuffer], nodeLowerBounds, nodeUpperBounds, keys,
            permutation, numBodies);
        thrust::sort_by_key(thrust::device, keys, keys + numBodies, permutation);
        LAUNCH(gatherKernel, numBodies, position[positionBuffer], velocity[velocityBuffer], id[idBuffer],
            permutation, position[1 - positionBuffer], velocity[1 - velocityBuffer], id[1 - idBuffer],
            numBodies);
        positionBuffer = 1 - positionBuffer;
        velocityBuffer = 1 - velocityBuffer;
        idBuffer = 1 - idBuffer;
    }
    void build()
    {
        LAUNCH(leafKernel, numLeaves, position[positionBuffer], nodeCenterOfMass, nodeOpeningDistanceSquared,
            nodeLowerBounds, nodeUpperBounds, numBodies, numLeaves, openingAngle);
        for (int first = numLeaves / 2; first >= 1; first /= 2)
        {
            LAUNCH(levelKernel, first, nodeCenterOfMass, nodeOpeningDistanceSquared, nodeLowerBounds,
                nodeUpperBounds, first, openingAngle);
        }
    }
    void treeStep()
    {
        LAUNCH(treeStepKernel, numBodies, position[positionBuffer], velocity[velocityBuffer],
            position[1 - positionBuffer], nodeCenterOfMass, nodeOpeningDistanceSquared, numBodies, numLeaves,
            timeStep, softening);
        positionBuffer = 1 - positionBuffer;
    }
    void directStep()
    {
        LAUNCH(directStepKernel, numBodies, position[positionBuffer], velocity[velocityBuffer],
            position[1 - positionBuffer], numBodies, timeStep, softening);
        positionBuffer = 1 - positionBuffer;
    }
};

int main(int argc, char** argv)
{
    double programStart = nowMs();
    int numBodies = (argc > 1) ? atoi(argv[1]) : NUM_BODIES;
    int numSteps = (argc > 2) ? atoi(argv[2]) : NUM_STEPS;
    float timeStep = (argc > 3) ? atof(argv[3]) : TIME_STEP;
    float gravitationalConstant = (argc > 4) ? atof(argv[4]) : GRAVITATIONAL_CONSTANT;
    float softening = (argc > 5) ? atof(argv[5]) : SOFTENING;
    int saveEvery = (argc > 6) ? atoi(argv[6]) : SAVE_EVERY;
    int scenario = (argc > 7) ? atoi(argv[7]) : SCENARIO;
    float openingAngle = (argc > 8) ? atof(argv[8]) : OPENING_ANGLE;
    const char* mode = (argc > 9) ? argv[9] : MODE; // "tree" or "direct"
    int resortEvery = (argc > 10) ? atoi(argv[10]) : RESORT_EVERY;
    int maxFrameBodies = (argc > 11) ? atoi(argv[11]) : MAX_FRAME_BODIES;
    int checkBodies = (argc > 12) ? atoi(argv[12]) : CHECK_BODIES; // 0 skips the force check.

    bool useTree = (strcmp(mode, "direct") != 0);
    if (openingAngle > 1.0f)
    {
        openingAngle = 1.0f; // The opening test is only safe up to 1.
    }
    if (openingAngle < 0.0f)
    {
        openingAngle = 0.0f;
    }
    if (resortEvery < 1)
    {
        resortEvery = 1;
    }
    if (saveEvery < 0)
    {
        saveEvery = 0;
    }
    if (numBodies < 2)
    {
        numBodies = 2;
    }

    Simulation simulation;
    simulation.numBodies = numBodies;
    simulation.openingAngle = openingAngle;
    simulation.softening = softening;
    simulation.timeStep = timeStep;
    int leaves = (numBodies + LEAF - 1) / LEAF;
    simulation.numLeaves = 1;
    while (simulation.numLeaves < leaves)
    {
        simulation.numLeaves *= 2;
    }
    int numNodes = 2 * simulation.numLeaves;

    size_t bodyBytes = (size_t)numBodies * sizeof(float4);
    float4* position = (float4*)malloc(bodyBytes);
    float4* velocity = (float4*)malloc(bodyBytes);
    float* mass = (float*)malloc((size_t)numBodies * sizeof(float));
    uint32_t* bodyId = (uint32_t*)malloc((size_t)numBodies * sizeof(uint32_t));
    if (scenario == 2)
    {
        initRing(position, velocity, numBodies);
    }
    else if (scenario == 1)
    {
        initGalaxies(position, velocity, numBodies, gravitationalConstant, softening);
    }
    else
    {
        initSphere(position, velocity, numBodies);
    }

    // Save every stride-th body in original order.
    int stride = (numBodies + maxFrameBodies - 1) / maxFrameBodies;
    if (stride < 1)
    {
        stride = 1;
    }
    int frameBodies = (numBodies + stride - 1) / stride;
    int numFrames = (saveEvery > 0) ? numSteps / saveEvery + 1 : 0;
    size_t frameBytes = (size_t)frameBodies * sizeof(float4);

    FILE* fbin = fopen(FRAMES_FILE, "wb");
    if (fbin == nullptr)
    {
        perror(FRAMES_FILE);
        return 1;
    }
    fwrite(&frameBodies, sizeof(int), 1, fbin);
    fwrite(&numFrames, sizeof(int), 1, fbin);
    if (saveEvery > 0)
    {
        for (int i = 0; i < numBodies; i += stride)
        {
            fwrite(&position[i], sizeof(float4), 1, fbin); // Initial frame.
        }
    }

    float4 boxLo = make_float4(EMPTY_BOX, EMPTY_BOX, EMPTY_BOX, 0.0f);
    float4 boxHi = make_float4(-EMPTY_BOX, -EMPTY_BOX, -EMPTY_BOX, 0.0f);
    for (int i = 0; i < numBodies; ++i)
    {
        mass[i] = position[i].w;
        position[i].w *= gravitationalConstant; // Multiply mass by G once.
        bodyId[i] = (uint32_t)i;
        boxLo.x = fminf(boxLo.x, position[i].x);
        boxHi.x = fmaxf(boxHi.x, position[i].x);
        boxLo.y = fminf(boxLo.y, position[i].y);
        boxHi.y = fmaxf(boxHi.y, position[i].y);
        boxLo.z = fminf(boxLo.z, position[i].z);
        boxHi.z = fmaxf(boxHi.z, position[i].z);
    }

    for (int b = 0; b < 2; ++b)
    {
        CUDA_CHECK(cudaMalloc(&simulation.position[b], bodyBytes));
        CUDA_CHECK(cudaMalloc(&simulation.velocity[b], bodyBytes));
        CUDA_CHECK(cudaMalloc(&simulation.id[b], (size_t)numBodies * sizeof(uint32_t)));
    }
    CUDA_CHECK(cudaMalloc(&simulation.keys, (size_t)numBodies * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&simulation.permutation, (size_t)numBodies * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&simulation.nodeCenterOfMass, (size_t)numNodes * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&simulation.nodeLowerBounds, (size_t)numNodes * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&simulation.nodeUpperBounds, (size_t)numNodes * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&simulation.nodeOpeningDistanceSquared, (size_t)numNodes * sizeof(float)));
    simulation.positionBuffer = simulation.velocityBuffer = simulation.idBuffer = 0;
    CUDA_CHECK(cudaMemcpy(simulation.position[0], position, bodyBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(simulation.velocity[0], velocity, bodyBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(simulation.id[0], bodyId, (size_t)numBodies * sizeof(uint32_t), cudaMemcpyHostToDevice));
    // The first sort needs a bounding box before the tree exists.
    CUDA_CHECK(cudaMemcpy(simulation.nodeLowerBounds + 1, &boxLo, sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(simulation.nodeUpperBounds + 1, &boxHi, sizeof(float4), cudaMemcpyHostToDevice));

    float4* gpuFrames = nullptr;
    float4* frames = nullptr;
    if (numFrames > 1)
    {
        CUDA_CHECK(cudaMalloc(&gpuFrames, (size_t)(numFrames - 1) * frameBytes));
        CUDA_CHECK(cudaHostAlloc(&frames, (size_t)(numFrames - 1) * frameBytes, cudaHostAllocDefault));
    }

    // Check the tree force against the direct force.
    if (useTree && checkBodies > 0)
    {
        int M = (checkBodies < numBodies) ? checkBodies : numBodies;
        float* d_err;
        int* d_cnt;
        CUDA_CHECK(cudaMalloc(&d_err, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cnt, M * sizeof(int)));
        simulation.sort();
        simulation.build();
        LAUNCH(checkKernel, M, simulation.position[simulation.positionBuffer], simulation.nodeCenterOfMass,
            simulation.nodeOpeningDistanceSquared, numBodies, simulation.numLeaves, softening, M, d_err,
            d_cnt);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> err(M);
        std::vector<int> cnt(M);
        CUDA_CHECK(cudaMemcpy(err.data(), d_err, M * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cnt.data(), d_cnt, M * sizeof(int), cudaMemcpyDeviceToHost));
        std::sort(err.begin(), err.end());
        double meanCnt = 0;
        for (int i = 0; i < M; ++i)
        {
            meanCnt += cnt[i];
        }
        meanCnt /= M;
        printf("checked bodies              : %d\n", M);
        printf("median relative force error : %.9g\n", err[M / 2]);
        printf("95th percentile force error : %.9g\n", err[(int)(0.95 * (M - 1))]);
        printf("max relative force error    : %.9g\n", err[M - 1]);
        printf("mean interactions per body  : %.9g\n", meanCnt);
        cudaFree(d_err);
        cudaFree(d_cnt);
    }

    // Wait after each phase to measure sort, build and force separately.
    double setupMs = nowMs() - programStart;
    double tSort = 0, tBuild = 0, tForce = 0;
    int frame = 0;
    CUDA_CHECK(cudaDeviceSynchronize());
    double t0 = nowMs();
    for (int step = 0; step < numSteps; ++step)
    {
        double a = nowMs();
        if (useTree)
        {
            if (step % resortEvery == 0)
            {
                simulation.sort();
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            double b = nowMs();
            simulation.build();
            CUDA_CHECK(cudaDeviceSynchronize());
            double c = nowMs();
            simulation.treeStep();
            CUDA_CHECK(cudaDeviceSynchronize());
            double d = nowMs();
            tSort += b - a;
            tBuild += c - b;
            tForce += d - c;
        }
        else
        {
            simulation.directStep();
            CUDA_CHECK(cudaDeviceSynchronize());
            tForce += nowMs() - a;
        }
        if (saveEvery > 0 && (step + 1) % saveEvery == 0 && frame < numFrames - 1)
        {
            LAUNCH(saveFrameKernel, numBodies, simulation.position[simulation.positionBuffer],
                simulation.id[simulation.idBuffer], gpuFrames + (size_t)frame * frameBodies, numBodies,
                stride);
            ++frame;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    double ms = nowMs() - t0;

    double saveStart = nowMs();
    if (frame > 0)
    {
        CUDA_CHECK(cudaMemcpy(frames, gpuFrames, (size_t)frame * frameBytes, cudaMemcpyDeviceToHost));
        fwrite(frames, sizeof(float4), (size_t)frame * frameBodies, fbin);
    }
    fclose(fbin);

    // Save the final state in original order, up to 100000 bodies.
    CUDA_CHECK(cudaMemcpy(
        position, simulation.position[simulation.positionBuffer], bodyBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        velocity, simulation.velocity[simulation.velocityBuffer], bodyBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bodyId, simulation.id[simulation.idBuffer], (size_t)numBodies * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));
    std::vector<int> where(numBodies);
    for (int i = 0; i < numBodies; ++i)
    {
        where[bodyId[i]] = i;
    }
    int rows = (numBodies < 100000) ? numBodies : 100000;
    FILE* fout = fopen(FINAL_STATE_FILE, "w");
    if (fout == nullptr)
    {
        perror(FINAL_STATE_FILE);
        return 1;
    }
    fprintf(fout, "i,x,y,z,vx,vy,vz,m\n");
    for (int i = 0; i < rows; ++i)
    {
        int k = where[i];
        fprintf(fout, "%d,%f,%f,%f,%f,%f,%f,%f\n", i, position[k].x, position[k].y, position[k].z,
            velocity[k].x, velocity[k].y, velocity[k].z, mass[i]);
    }
    fclose(fout);

    double saveMs = nowMs() - saveStart;
    double totalMs = nowMs() - programStart;
    printf("version                          : v4 Barnes-Hut GPU\n");
    printf("mode                             : %s\n", useTree ? "tree" : "direct");
    printf("random seed          : %u\n", (unsigned int)INITIAL_SEED);
    printf("bodies                           : %d\n", numBodies);
    printf("steps                            : %d\n", numSteps);
    printf("time step                        : %.9g\n", timeStep);
    printf("save every                       : %d\n", saveEvery);
    printf("gravitational constant           : %.9g\n", gravitationalConstant);
    printf("softening                        : %.9g\n", softening);
    printf("block size                       : %d\n", BLOCK_SIZE);
    printf("scenario                         : %d\n", scenario);
    printf("opening angle                    : %.9g\n", openingAngle);
    printf("resort every                     : %d\n", resortEvery);
    printf("leaf size                        : %d\n", LEAF);
    printf("tree leaves                      : %d\n", simulation.numLeaves);
    printf("tree nodes                       : %d\n", numNodes);
    printf("max frame bodies                 : %d\n", maxFrameBodies);
    printf("check bodies                     : %d\n", checkBodies);
    printf("setup time ms                    : %.3f\n", setupMs);
    printf("sort time ms                     : %.3f\n", tSort);
    printf("build time ms                    : %.3f\n", tBuild);
    printf("force time ms                    : %.3f\n", tForce);
    printf("sort ms per step                 : %.5f\n", tSort / numSteps);
    printf("build ms per step                : %.5f\n", tBuild / numSteps);
    printf("force ms per step                : %.5f\n", tForce / numSteps);
    printf("compute time ms                  : %.3f\n", ms);
    printf("saving time ms                   : %.3f\n", saveMs);
    printf("total time ms                    : %.3f\n", totalMs);
    printf("ms per step                      : %.5f\n", ms / numSteps);
    // This rate uses N squared, not the actual number of tree interactions.
    // G is already in mass, so addForce uses 19 operations when rsqrtf counts as one.
    printf("all-pairs equivalent billion interactions per sec: %.3f\n",
        (double)numBodies * (double)numBodies * numSteps / (ms * 1.0e6));
    printf("Saved %d frames of %d bodies to %s; final state of %d bodies to %s\n", frame + 1, frameBodies,
        FRAMES_FILE, rows, FINAL_STATE_FILE);

    for (int b = 0; b < 2; ++b)
    {
        cudaFree(simulation.position[b]);
        cudaFree(simulation.velocity[b]);
        cudaFree(simulation.id[b]);
    }
    cudaFree(simulation.keys);
    cudaFree(simulation.permutation);
    cudaFree(simulation.nodeCenterOfMass);
    cudaFree(simulation.nodeLowerBounds);
    cudaFree(simulation.nodeUpperBounds);
    cudaFree(simulation.nodeOpeningDistanceSquared);
    if (gpuFrames)
    {
        cudaFree(gpuFrames);
    }
    if (frames)
    {
        cudaFreeHost(frames);
    }
    free(position);
    free(velocity);
    free(mass);
    free(bodyId);
    return 0;
}
