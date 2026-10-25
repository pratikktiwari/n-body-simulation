# CPU simulation

[cpu.ipynb](./cpu.ipynb) is the direct CPU version and the starting point for the
CUDA implementations, since its pair loop makes it easy to follow how one
body's acceleration becomes a velocity and position update. The shared
[physics and formulas](../README.md#what-the-simulation-calculates) are in the
parent README, along with the [notebook workflow](../README.md#running-the-notebooks)
and [output formats](../README.md#saved-results).

## What this version does

The simulation runs on one CPU thread, with an outer loop for the body being
updated and an inner loop over all source bodies. Each body's acceleration
starts at zero, `addForce` adds each source body's contribution through a
reference parameter, and the completed acceleration is used to update velocity
and write the next position.

`Body` stores x, y, z and mass, while `Velocity` stores three components and is
also used for the temporary acceleration value. The two quantities are
physically different even though they share the same container, and separate
position buffers keep the old state available until all bodies finish the step.

This version uses `sqrtf` followed by division for the acceleration scale,
rather than the reciprocal-square-root multiplication used by CUDA. There are
no GPU transfers or parallel reductions, which makes it a useful reference for
understanding the later versions, although floating-point results need not match
them exactly.

## Build and run

Generate `nbody_cpu.cpp` from the notebook's source cell before compiling. With
the default output filenames, the terminal equivalent of the compile/run cells
is:

```bash
g++ -O3 -o nbody_cpu.out nbody_cpu.cpp \
  '-DPOSITIONS_FILE="positions.csv"' '-DFINAL_STATE_FILE="final_state.csv"'
./nbody_cpu.out 3296
```

The notebook's main run supplies 3,296 bodies, while running the executable
without arguments uses `NUM_BODIES`; the remaining argument order is the
[shared direct-version order](../README.md#running-the-notebooks).

## What the measurements include

Compute time is the accumulated wall time around the simulation step and
position-buffer swap, while saving time covers frame CSV writes. Total time
also includes setup, final-state output and cleanup, and the GPU time field is
zero because this implementation does not use a GPU.

The scaling sweep uses 16, 64, 256, 1,024, 2,048, 4,096 and 10,000 bodies,
inheriting the source's other defaults, including 1,000 steps and frame saving.
That is a different step count from the CUDA sweeps, so compare per-step compute
time or match the settings before comparing total runtime; buffered writes also
mean the saving timer is not a pure measurement of disk speed.
