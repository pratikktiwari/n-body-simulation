# Tiled CUDA simulation

[tiled.ipynb](./tiled.ipynb) keeps the naive version's one-thread-per-body
calculation, but shares source bodies through a block's shared memory instead
of having every thread read them separately from global memory. The
[physics and update rule](../README.md#what-the-simulation-calculates) stay the
same, so this version changes how data is reused rather than how many real
pairs are needed.

## Why use a shared tile

A block loads one group of source bodies into a shared array, then all its
threads use that group before moving to the next one. With the default
`BLOCK_SIZE` of 256, each thread loads one source body and uses the whole
256-body tile for its target body's acceleration.

The first barrier makes sure the tile is ready before it is read, while the
second stops a fast thread from overwriting data that another thread still
needs. This is why extra target threads cannot return early as they do in the
naive version: they must reach both barriers, even though only real bodies
write an updated state.

Extra source slots in the last tile have zero mass, so they add no force.
Padding makes arbitrary body counts work without a separate final-tile kernel,
although it also means some executed work is not included in the reported
N-squared pair count.

## Build and run

Generate `nbody_tiled.cu` from the notebook before compiling. The command below
uses the default output filenames and keeps the same compiler flags as naive
CUDA:

```bash
nvcc -O3 -arch=sm_75 -o nbody_tiled.out nbody_tiled.cu \
  '-DPOSITIONS_FILE="positions.csv"' '-DFINAL_STATE_FILE="final_state.csv"'
./nbody_tiled.out
```

The launch and tile size both use `BLOCK_SIZE`, while the notebook reads its
runtime settings from [comparison_config.json](../comparison_config.json) and
passes the [shared direct-version arguments](../README.md#running-the-notebooks)
explicitly.
The position/mass and velocity layouts remain `float4` and `float3`, with the
same reciprocal-square-root force calculation as naive CUDA.

## What to compare

CUDA events measure each update kernel, and frame transfers back to the CPU
plus CSV writes are timed separately as saving. The scaling sweep runs the
shared body counts for 100 steps with frame saving disabled, so its compute
chart can be compared with naive CUDA's kernel time under matching inputs.
Benchmark runs use temporary folders and keep the main frames and final state
unchanged; `run.json` and `scaling.json` record settings and provenance for
[compare.ipynb](../compare.ipynb), which also displays the main GIF.

Tiling still has quadratic pair work and adds barriers, so the benefit depends
on body count, memory reuse and the GPU rather than being a guaranteed speedup
for every N. The [common output and performance sections](../README.md#saved-results)
cover the CSVs, GIF and estimated throughput without repeating them here.
