#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <utility>

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

struct Body
{
    float x;    // position x
    float y;    // position y
    float z;    // position z
    float mass; // mass of the body
};

struct Velocity
{
    float x; // velocity in x direction
    float y; // velocity in y direction
    float z; // velocity in z direction
};

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Start bodies in a ring with random positions and sideways motion.
void initBodies(Body* pos, Velocity* vel, int numBodies)
{
    // Same seed for repeatable starting values.
    srand(INITIAL_SEED);

    for (int i = 0; i < numBodies; i++)
    {
        float radius = 1.0f + (rand() / (float)RAND_MAX) * 3.0f;
        float angle = (rand() / (float)RAND_MAX) * 2.0f * M_PI;

        pos[i].x = radius * cosf(angle);
        pos[i].y = radius * sinf(angle);
        pos[i].z = ((rand() / (float)RAND_MAX) - 0.5f) * 0.5f;
        pos[i].mass = 1.0f + (rand() / (float)RAND_MAX) * 5.0f;

        float speed = 0.5f * sqrtf(1.0f / radius);
        vel[i].x = -speed * sinf(angle);
        vel[i].y = speed * cosf(angle);
        vel[i].z = 0.0f;
    }
}

// Add body2's pull to body1's acceleration.
// Softening keeps the force finite when bodies get very close.
void addForce(const Body& body1, const Body& body2, Velocity& acceleration, float gravitationalConstant,
    float softening)
{
    float deltaX = body2.x - body1.x;
    float deltaY = body2.y - body1.y;
    float deltaZ = body2.z - body1.z;

    // √((x₂ - x₁)² + (y₂ - y₁)² + (z₂ - z₁)² + softening)
    float distance = sqrtf((deltaX * deltaX) + (deltaY * deltaY) + (deltaZ * deltaZ) + softening);

    // G * m₂ / distance³
    float accelerationScale = gravitationalConstant * body2.mass / (distance * distance * distance);

    acceleration.x += deltaX * accelerationScale;
    acceleration.y += deltaY * accelerationScale;
    acceleration.z += deltaZ * accelerationScale;
}

// Calculate accelerations, then update velocities and positions.
void simulateStep(const Body* position, Velocity* velocity, Body* newPosition, int numBodies, float timeStep,
    float gravitationalConstant, float softening)
{
    for (int i = 0; i < numBodies; i++)
    {
        Velocity acceleration = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < numBodies; j++)
        {
            addForce(position[i], position[j], acceleration, gravitationalConstant, softening);
        }

        velocity[i].x += acceleration.x * timeStep;
        velocity[i].y += acceleration.y * timeStep;
        velocity[i].z += acceleration.z * timeStep;

        newPosition[i] = position[i];
        newPosition[i].x += velocity[i].x * timeStep;
        newPosition[i].y += velocity[i].y * timeStep;
        newPosition[i].z += velocity[i].z * timeStep;
    }
}

// Save this step's positions to the CSV file.
void saveFrame(FILE* file, int step, const Body* pos, int numBodies)
{
    for (int i = 0; i < numBodies; i++)
    {
        fprintf(file, "%d,%d,%f,%f,%f,%f\n", step, i, pos[i].x, pos[i].y, pos[i].z, pos[i].mass);
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
    // ./nbody_cpu [bodies] [steps] [timeStep] [saveEvery] [gravitationalConstant] [softening]

    int numBodies = (argc > 1) ? atoi(argv[1]) : NUM_BODIES;
    int numSteps = (argc > 2) ? atoi(argv[2]) : NUM_STEPS;
    float timeStep = (argc > 3) ? atof(argv[3]) : TIME_STEP;
    int saveEvery = (argc > 4) ? atoi(argv[4]) : SAVE_EVERY; // 0 means no frames are saved.
    float gravitationalConstant = (argc > 5) ? atof(argv[5]) : GRAVITATIONAL_CONSTANT;
    float softening = (argc > 6) ? atof(argv[6]) : SOFTENING;

    // Start the wall and CPU timers.
    double programStart = nowMs();
    clock_t cpuStart = clock();

    // Allocate space for the bodies.
    Body* position = new Body[numBodies];
    Body* newPosition = new Body[numBodies];
    Velocity* velocity = new Velocity[numBodies];
    initBodies(position, velocity, numBodies);

    // Open the positions file.
    FILE* framesFile = fopen(POSITIONS_FILE, "w");
    fprintf(framesFile, "step,id,x,y,z,mass\n");

    // Record setup time.
    double setupMs = nowMs() - programStart;
    double computeMs = 0.0;
    double saveMs = 0.0;
    if (saveEvery > 0)
    {
        double saveStart = nowMs();
        saveFrame(framesFile, 0, position, numBodies);
        saveMs += nowMs() - saveStart;
    }

    // Run the simulation.
    for (int step = 0; step < numSteps; step++)
    {
        double start = nowMs();
        simulateStep(position, velocity, newPosition, numBodies, timeStep, gravitationalConstant, softening);
        std::swap(position, newPosition);
        computeMs += nowMs() - start;

        if (saveEvery > 0 && (step + 1) % saveEvery == 0)
        {
            start = nowMs();
            saveFrame(framesFile, step + 1, position, numBodies);
            saveMs += nowMs() - start;
        }
    }
    fclose(framesFile);

    FILE* finalFile = fopen(FINAL_STATE_FILE, "w");
    fprintf(finalFile, "id,x,y,z,vx,vy,vz,mass\n");
    for (int i = 0; i < numBodies; i++)
    {
        fprintf(finalFile, "%d,%f,%f,%f,%f,%f,%f,%f\n", i, position[i].x, position[i].y, position[i].z,
            velocity[i].x, velocity[i].y, velocity[i].z, position[i].mass);
    }

    fclose(finalFile);

    delete[] position;
    delete[] newPosition;
    delete[] velocity;

    double totalMs = nowMs() - programStart;
    double cpuMs = 1000.0 * (clock() - cpuStart) / CLOCKS_PER_SEC;
    printReport("v1 CPU", numBodies, numSteps, setupMs, computeMs, 0.0, saveMs, totalMs, cpuMs, timeStep,
        saveEvery, gravitationalConstant, softening);
    return 0;
}
