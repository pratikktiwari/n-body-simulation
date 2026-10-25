# Barnes-Hut CUDA simulation

[barnes_hut.ipynb](./barnes_hut.ipynb) reduces the number of force interactions by
using a group's centre of mass when it is far enough away, rather than
evaluating every source body separately. It uses the
[same softened gravity and velocity-first update](../README.md#what-the-simulation-calculates),
but the approximation makes it a different approach from the direct CUDA
versions, not just another way to schedule the same pair loop.

## Sorting and building the tree

Morton keys put nearby bodies close together in memory, and Thrust sorts the
keys while the matching positions, velocities and original IDs move together.
Leaves then cover consecutive groups of sorted bodies, with a padded
power-of-two leaf count and parents built from their two children's masses
and bounding boxes.

The tree is an implicit binary tree stored in arrays, with root 1 and children
`2*k` and `2*k+1`, rather than a pointer-based octree. Each body walks this tree
without a stack, opening nearby nodes and using a node's total mass and centre
of mass when the opening test allows it; sorting happens every `RESORT_EVERY`
steps, while the tree is rebuilt from current positions every step.

The opening test accepts a nonempty node when its centre-of-mass distance d
exceeds:

$$
d>\frac{s}{\theta}+\delta
$$

Here s is the longest side of the node's bounding box, theta is the opening
angle, and delta is the offset between its centre of mass and box centre.
Smaller angles open more nodes and usually improve accuracy at the cost of
more work, while an angle of zero opens nonempty nodes at the scales used here.
The work is often closer to N log N, but can still approach quadratic work
when many nodes need to be opened.

## Scenes and tree settings

Unlike the ring versions, this notebook starts with two tilted rotating
galaxies by default, with a cold uniform sphere available as scenario 0.
Galaxy disk masses scale with body count and starting orbital speeds depend
on G, so matching the shared constants does not make these scenes equivalent
to the ring or guarantee the same visible motion.

| Setting | Default | Purpose |
|---|---:|---|
| `SCENARIO` | 1 | Two galaxies; 0 selects the sphere |
| `OPENING_ANGLE` | 0.6 | Controls how readily a group is accepted |
| `MODE` | `tree` | `direct` uses the reference all-pairs kernel |
| `LEAF` | 16 | Maximum bodies per leaf, set at compile time |
| `RESORT_EVERY` | 4 | Steps between Morton sorts |
| `MAX_FRAME_BODIES` | 32768 | Limit used to choose frame sampling stride |
| `CHECK_BODIES` | 1024 | Number of bodies requested for the force check |

The common physical defaults are listed in the [parent README](../README.md#starting-conditions-and-units).
The opening angle is clamped to zero through one, body count to at least two,
and sort/save intervals to at least one, which means **zero does not disable
frame saving here**. A force-check count of zero skips that check, and only
the exact mode string `direct` selects the direct kernel.

## Build and run

Generate `nbody_tree.cu` from the notebook, then compile with CUDA and Thrust
available. The renderer also needs SciPy in addition to the shared Python
packages:

```bash
nvcc -O3 -arch=sm_75 --use_fast_math nbody_tree.cu -o nbody_tree.out \
  '-DFRAMES_FILE="frames.bin"' '-DFINAL_STATE_FILE="final_output.csv"'
./nbody_tree.out
```

The argument order differs from the direct versions, so gravity and softening
come before the save interval:

```text
./nbody_tree.out [bodies] [steps] [timeStep] [gravitationalConstant] [softening] [saveEvery] [scenario] [openingAngle] [mode] [resortEvery] [maxFrameBodies] [checkBodies]
```

Use a positive maximum frame-body count and a nonnegative force-check count,
along with the positive step count, time step and softening described in the
parent guide. `LEAF` can be overridden with `-DLEAF=...`, which the notebook's
leaf experiment uses to create separate `.out` executables.

## Frames and final state

The filename constants select `frames.bin`, `final_output.csv`, `scaling.csv`,
`accuracy.csv`, `leaf_scaling.csv` and `nbody.gif`. Binary frames keep animation
output smaller than a full positions CSV, and sample every
`ceil(N / maxFrameBodies)`-th body in original ID order.

The file starts with two native 32-bit integers for saved-body count and frame
count, followed by `float4` records for the initial frame and each saving
interval. Mass is multiplied by G once on the CPU for the force kernels, so
the first frame contains original masses while later records hold `G * mass`
in w; the renderer uses the first frame for brightness and later xyz values
for motion.

The default run saves 501 frames of 10,000 bodies, roughly 80 MB, with later
frames buffered on the GPU before a bulk copy to pinned host memory. The GIF
uses an orbiting camera, density glow and fading trails, while the final CSV
keeps original masses and body order with columns `i,x,y,z,vx,vy,vz,m`, capped
at the first 100,000 bodies.

## Accuracy and performance experiments

The force check compares tree and direct accelerations on sampled bodies
before the main simulation, reporting median, 95th-percentile and maximum
relative error along with mean traversal interactions. It checks the starting
state rather than bounding error over the whole trajectory, and its work is
included in setup time.

Sort, build and force phases use synchronized wall timers, while compute time
covers the whole simulation loop, including frame capture and synchronization.
Saving time covers the later frame transfer and final-state writes, and total
time runs from main entry to reporting, before cleanup.

The scaling sweep measures tree runs for 20 steps and direct runs for five
steps up to 262,144 bodies, then labels larger direct times as projections
from the last measured rate. Benchmarks use a save interval longer than the
run so only the initial frame is written, and the plots compare per-step time,
tree phases and the explicitly labeled all-pairs-equivalent rate.

The angle sweep checks accuracy at different opening angles, while the leaf
sweep recompiles with 8, 16, 32 and 64 bodies per leaf to compare the balance
between tree depth and direct sums. Their results are exported separately,
and the parent guide explains why the
[equivalent rate is not actual tree GFLOPS](../README.md#reading-the-performance-results).
