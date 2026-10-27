# Optimised CUDA simulation

[optimised.ipynb](./optimised.ipynb) splits each target body's force sum across
several threads, which gives the GPU more work to run in parallel when the body
count is small. It is still direct all-pairs rather than Barnes-Hut, using the
same [ring, gravity and update rule](../README.md#what-the-simulation-calculates)
as the other direct versions.

## Splitting the force sum

A block works on 32 target bodies, with the lane identifying the body and each
warp handling a different slice of source bodies. Threads within a warp read
the same source body through the cache, so this version does not use the tiled
version's shared array of source positions.

Each thread writes its partial acceleration to shared memory, then a barrier
allows the first warp to add the slices and update the bodies. The block size is
therefore `32 * numSlices`, not a fixed 256 threads, and the report records both
the chosen slice count and actual block size.

Automatic selection tries 1, 2, 4, 8, 16 and 32 slices on copied device buffers,
using a warm-up followed by ten timed steps for each choice. The fastest option
is kept without advancing the main simulation, and the tuning cost appears in
setup time rather than the main compute measurement.

## Build and run

After generating `nbody_opt.cu`, the default compile/run commands are:

```bash
nvcc -O3 -arch=sm_75 --use_fast_math -o nbody_opt.out nbody_opt.cu \
  '-DPOSITIONS_FILE="positions.csv"' '-DFINAL_STATE_FILE="final_state.csv"'
./nbody_opt.out
```

This version adds a final argument to the shared direct-version settings:

```text
./nbody_opt.out [bodies] [steps] [timeStep] [saveEvery] [gravitationalConstant] [softening] [slices]
```

The notebook reads [comparison_config.json](../comparison_config.json) and
passes every argument explicitly, including the requested slice selection.
Its main ring, body count and physical inputs match the other variants.

`NUM_SLICES` defaults to zero, and a nonpositive runtime value enables automatic
selection; an explicit value should be one of the supported slice counts above.
The `--use_fast_math` flag and the changed summation order can affect rounding,
so bit-for-bit agreement with the CPU, naive or tiled version is not expected.

## Keeping frames on the GPU

Instead of waiting after each step to copy a frame to the CPU, this version
queues updates and device-to-device frame copies, then transfers all saved
frames back together after the simulation. This reduces repeated host waits,
but needs a GPU buffer proportional to body count and saved-frame count.

With 10,000 bodies, 1,000 steps and a frame every two steps, that frame buffer
alone needs about 80 MB. Larger runs can use much more memory, while setting
`saveEvery` to zero removes frame buffering but still writes the final state.

## What the timing means

The GPU/compute timer surrounds the whole queued sequence, so it includes
device-to-device frame copies when saving is enabled, unlike naive and tiled
CUDA's per-kernel measurements. Saving time covers the later bulk transfer and
frame CSV writes, while total runtime also includes tuning, setup, final-state
output and cleanup.

The scaling sweep uses the shared body counts for 100 steps with frame saving
disabled, so buffered frame copies are outside that compute measurement.
Temporary benchmark folders preserve the main outputs, while `run.json` and
`scaling.json` record settings and provenance for
[compare.ipynb](../compare.ipynb). The main GIF has the same view and playback
rate as the other variants, and the shared
[performance formulas](../README.md#reading-the-performance-results) still use
the nominal direct pair count and exclude reduction overhead from the operation
estimate.
