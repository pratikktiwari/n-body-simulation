# Naive CUDA simulation

[naive.ipynb](./naive.ipynb) moves the direct pair calculation onto the GPU without
adding shared-memory tiles or splitting a body's work across threads. It uses
the same ring and update rule as the CPU version, with the
[physics, formulas and common workflow](../README.md) covered in the parent
README.

## What changes on the GPU

Each GPU thread handles one target body and loops over all source bodies,
reading their positions from global memory. This makes different target bodies
run in parallel, but every thread still has a full source loop, so the total
pair work remains quadratic.

Positions and mass use CUDA's `float4`, with mass in the w component, while
velocity and acceleration use `float3`. The force helper uses `rsqrtf` and
multiplies the inverse distance three times, rather than using the CPU version's
square root and direct division.

The default block size is 256, with enough blocks to cover all bodies and an
early return for extra threads in the last block. A thread writes only its own
velocity and new position, so it does not need to coordinate with other threads
inside the kernel; the separate position buffers keep its source reads on the
old state.

## Build and run

Generate `nbody_naive.cu` from the source cell, then compile it with an NVIDIA
CUDA toolkit. The notebook uses the following settings, shown here with default
output filenames:

```bash
nvcc -O3 -arch=sm_75 -o nbody_naive.out nbody_naive.cu \
  '-DPOSITIONS_FILE="positions.csv"' '-DFINAL_STATE_FILE="final_state.csv"'
./nbody_naive.out 10000 1000 0.005 2 0.001 0.01
```

The notebook takes its main and benchmark inputs from
[comparison_config.json](../comparison_config.json), passing every runtime
setting explicitly. `BLOCK_SIZE` is a compile-time setting rather than an
extra command-line argument, and the arguments follow the
[shared direct-version order](../README.md#running-the-notebooks).

## Transfers and timing

CUDA events measure each update kernel, after which the CPU waits for completion
and swaps the position buffers. When a frame is due, positions are copied back
to the CPU and written to CSV, so the GPU/compute timer excludes those frame
copies and writes while the saving timer measures them separately.

Setup includes the first CUDA startup, allocations and initial transfers, while
total wall time includes synchronization, saving, final-state output and
cleanup. Small runs can spend more time on startup and launches than useful
parallel work, so a low body count does not necessarily show the benefit of
using a GPU.

The scaling sweep uses the shared body counts for 100 steps without saving
frames, so its inputs match the other variants. Temporary benchmark folders
keep the main frames and final state intact, and saved exports in `run.json`
and `scaling.json` let [compare.ipynb](../compare.ipynb) load the timings and GIF
without running another simulation. The shared
[performance guide](../README.md#reading-the-performance-results) explains the
timing boundaries and throughput estimates.
