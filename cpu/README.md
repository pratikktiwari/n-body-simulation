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
./nbody_cpu.out 10000 1000 0.005 2 0.001 0.01
```

The notebook reads [comparison_config.json](../comparison_config.json), passes
every run argument explicitly and exports results through
[benchmark.py](../benchmark.py). The example above matches its current main run;
the remaining argument order is the
[shared direct-version order](../README.md#running-the-notebooks).

## What the measurements include

Compute time is the accumulated wall time around the simulation step and
position-buffer swap, while saving time covers frame CSV writes. Total time
also includes setup, final-state output and cleanup, and the GPU time field is
zero because this implementation does not use a GPU.

The scaling sweep uses the common body counts, including 8,192, with 100 steps
and frame saving disabled, so its inputs match the CUDA sweeps. Benchmark files
are written in temporary folders rather than replacing the main outputs, while
`scaling.csv` and `scaling.json` keep the measurements and provenance for
[compare.ipynb](../compare.ipynb). The main run's `run.json` also records its
settings and output checksums, and its GIF uses the same view and playback rate
as the other variants.
