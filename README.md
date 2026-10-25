# N-Body Simulation

This project follows the same gravity simulation from a CPU implementation to
several CUDA implementations, so it is easier to see which changes come from
parallel execution, which come from memory access, and which come from doing
less work. The direct versions calculate every pair of bodies, while Barnes-Hut
groups distant bodies and trades some accuracy for fewer interactions.

The physics and common notebook workflow are explained here, while each folder's
README covers the part that makes that implementation different.

| Implementation | Main idea | Notebook and details |
|---|---|---|
| CPU | Direct all-pairs calculation on one CPU thread | [Notebook](./cpu/cpu.ipynb) / [README](./cpu/README.md) |
| Naive CUDA | One GPU thread per body, reading source bodies from global memory | [Notebook](./gpu_naive/naive.ipynb) / [README](./gpu_naive/README.md) |
| Tiled CUDA | Share source bodies through a block's shared memory | [Notebook](./gpu_tiled/tiled.ipynb) / [README](./gpu_tiled/README.md) |
| Optimised CUDA | Split each body's force calculation across several threads | [Notebook](./gpu_optimised/optimised.ipynb) / [README](./gpu_optimised/README.md) |
| Barnes-Hut CUDA | Replace distant groups with their centre of mass | [Notebook](./gpu_barnes_hut/barnes_hut.ipynb) / [README](./gpu_barnes_hut/README.md) |

The [live interactive visualization](https://pratikktiwari.github.io/n-body-simulation/)
can be opened in a browser to inspect the ring's starting coordinates, experiment
with a single-body orbit and run a small live N-body system. It runs locally
without a build step or external libraries, with its HTML, styles and JavaScript
kept in separate files; the single-orbit demonstration uses velocity-Verlet,
while the N-body demo uses the notebooks' softened gravity and semi-implicit
Euler update, with 1 to 400 bodies to keep the direct browser calculation
manageable. To run it locally, open [simulation/index.html](./simulation/index.html)
directly in a browser.

## How the GPU implementations differ

Naive, tiled and optimised CUDA all calculate every pair of bodies, so their
differences come from how the GPU divides the work and reuses data rather than
from reducing the number of interactions. Barnes-Hut changes that part of the
calculation by approximating the pull of distant groups.

| Version | How it works | What it improves | Work per simulation step |
|---|---|---|---|
| Naive | One thread handles one target body and loops over all source bodies, reading their positions from global memory. | Provides a straightforward GPU version of the direct calculation. | O(N²) |
| Tiled | Each thread still handles one target body, but the block loads groups of source positions into shared memory for its threads to reuse. | Reduces repeated reads from global memory without changing which body pairs are calculated. | O(N²) |
| Optimised | Several threads calculate different parts of a target body's force sum, then combine their partial accelerations through shared memory. | Gives the GPU more parallel work when one thread per body is not enough to keep it busy. | O(N²) |
| Barnes-Hut | Builds a spatial tree and uses a distant group's total mass and centre of mass instead of calculating each body's pull separately. | Reduces the number of force calculations, with an accuracy trade-off controlled by the opening angle. | Often closer to O(N log N), but can approach O(N²). |

The optimised version is not simply the tiled version with more changes: tiled
shares source positions, while optimised shares partial force sums. Optimised
also tests different ways of splitting the work and buffers saved frames on the
GPU before copying them back together, so its tuning cost, memory use and timing
boundaries differ from naive and tiled.

These changes do not guarantee that each version is faster than the previous
one for every body count, since synchronization, reductions and tree construction
also take time. Comparisons need matching starting conditions and saving
settings, along with compatible timing boundaries; Barnes-Hut starts with a
different scene by default, and a smaller opening angle generally means more
work in exchange for better force accuracy.

## What the simulation calculates

Each body has a position, mass and velocity, while acceleration is calculated
again at every time step because the bodies keep moving. For body i, these are
written as:

$$
\mathbf{r}_i=(x_i,y_i,z_i),\qquad
m_i,\qquad
\mathbf{v}_i=(v_{x,i},v_{y,i},v_{z,i})
$$

Newton's law gives the force between two point masses separated by distance r:

$$
F=G\frac{m_i m_j}{r^2},\qquad
a_i=\frac{F}{m_i}=G\frac{m_j}{r^2}
$$

Since force is mass times acceleration, dividing by the mass of the body being
moved leaves the other body's mass in the acceleration calculation. This is why
`addForce` uses the source body's mass rather than multiplying both masses;
despite its name, the function adds acceleration, not force.

The displacement from body i towards body j gives the direction of that pull:

$$
\Delta\mathbf{r}_{ij}=\mathbf{r}_j-\mathbf{r}_i,\qquad
r^2=\Delta x^2+\Delta y^2+\Delta z^2
$$

Combining the inverse-square magnitude with the direction gives:

$$
\mathbf{a}_{ij}
=\frac{Gm_j}{r^2}\frac{\Delta\mathbf{r}_{ij}}{r}
=\frac{Gm_j\,\Delta\mathbf{r}_{ij}}{r^3}
$$

The extra division by distance turns displacement into a unit direction, so the
cubed denominator does not make gravity an inverse-cube magnitude law.

### Why softening is added

Two bodies can get very close, and a body also appears in its own pair loop, so
using distance alone could produce a zero denominator or a very large
acceleration. The code adds a small squared softening length before calculating
the distance, giving the acceleration used by the simulation:

$$
d^2=r^2+\varepsilon^2,\qquad
\mathbf{a}_{ij}
=\frac{Gm_j\,\Delta\mathbf{r}_{ij}}
       {(r^2+\varepsilon^2)^{3/2}}
$$

The setting called `SOFTENING` already stores epsilon squared, which means a value
of `0.01` gives a softening length of `0.1`. This changes gravity at short
distances and helps avoid extreme updates, although it cannot make an overly
large time step accurate.

For direct all-pairs calculations, the contributions add up as:

$$
\mathbf{a}_i=\sum_{j=0}^{N-1}
\frac{Gm_j(\mathbf{r}_j-\mathbf{r}_i)}
     {(\|\mathbf{r}_j-\mathbf{r}_i\|^2+\varepsilon^2)^{3/2}}
$$

The self-pair contributes zero when softening is positive because its
displacement is zero. Barnes-Hut uses the same softened interaction, but some
source bodies are represented by a group's total mass and centre of mass.

### Moving the bodies

Every version updates velocity before position, using semi-implicit Euler:

$$
\mathbf{v}_i^{n+1}=\mathbf{v}_i^n+\mathbf{a}_i^n\Delta t
$$

$$
\mathbf{r}_i^{n+1}=\mathbf{r}_i^n+\mathbf{v}_i^{n+1}\Delta t
$$

The new velocity is used immediately, but source positions stay unchanged until
the step is complete. Separate input and output position buffers make sure one
body does not read another body's partially updated position.

A smaller time step usually improves accuracy, but needs more steps to cover the
same simulated duration. This is still a first-order method rather than an exact
solution, so energy can drift and bodies do not merge or bounce when they meet.

## Starting conditions and units

The CPU, naive, tiled and optimised versions start with the same thin ring. For
independent random samples u between approximately zero and one, the starting
positions and masses are:

$$
R=1+3u_R,\qquad \theta=2\pi u_\theta
$$

$$
x=R\cos\theta,\qquad y=R\sin\theta,\qquad
z=0.5(u_z-0.5),\qquad m=1+5u_m
$$

Bodies start moving sideways around the origin, with:

$$
v_{\mathrm{initial}}=0.5\sqrt{\frac{1}{R}},\qquad
v_x=-v_{\mathrm{initial}}\sin\theta,\qquad
v_y=v_{\mathrm{initial}}\cos\theta,\qquad v_z=0
$$

These starting velocities are chosen to show motion rather than solve for a
stable orbit in the combined gravitational field, and they do not adjust when G
changes. A fixed seed repeats the starting values on the same C library, but
macOS and Linux do not necessarily produce the same random sequence.

Barnes-Hut has separate sphere and galaxy scenes, with the galaxies selected by
default, so matching its constants with a ring simulation does not make the
initial conditions equivalent.

The current source defaults are shared across the five implementations:

| Constant | Default | Meaning |
|---|---:|---|
| `NUM_BODIES` | 10000 | Number of bodies |
| `NUM_STEPS` | 1000 | Number of updates |
| `TIME_STEP` | 0.005 | Simulated time per update |
| `SAVE_EVERY` | 2 | Frame interval, with version-specific handling of zero |
| `GRAVITATIONAL_CONSTANT` | 0.001 | Gravity strength in simulation units |
| `SOFTENING` | 0.01 | Squared softening length |

These values use simulation units rather than SI gravity. The physical value of
G is approximately `6.67430e-11` in metres, kilograms and seconds, but using it
with the small masses and distances here would make attraction much weaker.
Changing G changes the motion, not just the time taken to calculate it.

Simulated duration is separate from execution time:

$$
T=S\Delta t=1000\times0.005=5
$$

This is five simulated time units, regardless of how long the computer takes.
The CPU and naive notebooks currently override the main run's body count to
3,296; omitted command-line arguments still come from the source constants.

## Running the notebooks

The CPU notebook needs a C++ compiler, while the CUDA notebooks need an NVIDIA
GPU, a compatible driver and the CUDA toolkit with `nvcc`. Apple Silicon cannot
run these CUDA kernels. Python 3.10+, Jupyter/IPython, NumPy, pandas, Matplotlib
and Pillow are used for the notebook workflow, with SciPy additionally used by
the Barnes-Hut renderer.

Start the kernel in the implementation's folder and run the cells in order:
platform information, filename constants, source generation, compilation, the
main simulation, final-state inspection, animation, then benchmarks. Animation
comes before benchmarking because each executable run overwrites the simulation
output, which would otherwise replace the main run's frames before they are read.

The constants cell below platform information controls saved filenames, while
source and executable names stay fixed. The compilation cell passes C++ output
filenames as quoted compiler definitions, so changing one of those filenames
requires recompilation before running again.

The compile commands use `-O3` for optimization, and CUDA commands currently
target `sm_75`, as used by an NVIDIA T4. Optimised CUDA and Barnes-Hut also use
`--use_fast_math`, which can change rounding and accuracy; choose a GPU target
supported by the installed toolkit rather than assuming the target fits every
machine. The notebook shell commands assume a POSIX-style environment such as
Linux, Colab or WSL.

For the four direct versions, arguments follow this order:

```text
./<executable>.out [bodies] [steps] [timeStep] [saveEvery] [gravitationalConstant] [softening]
```

Optimised CUDA adds a final slice-count argument, while Barnes-Hut has a different
argument order documented in its own README. Supply positive body and step
counts, time step and softening, since the existing `atoi`/`atof` parsing does not
safely reject malformed input.

## Saved results

The direct versions use the following folder-local files, so version suffixes
are not needed:

| Constant | File | Contents |
|---|---|---|
| `POSITIONS_FILE` | `positions.csv` | Saved frames: `step,id,x,y,z,mass` |
| `FINAL_STATE_FILE` | `final_state.csv` | Final state: `id,x,y,z,vx,vy,vz,mass` |
| `SCALING_FILE` | `scaling.csv` | One row of settings and measurements per benchmark run |
| `GIF_FILE` | `nbody.gif` | Animation of the main run |

Frames are saved after updates labeled `0, SAVE_EVERY, ...`, so label zero is
not the starting state; its simulated time is given by:

$$
t_{\mathrm{frame}}=(\mathrm{step}+1)\Delta t
$$

Zero disables frame rows in the direct versions, although a header-only positions
file is still created and cannot be animated. Barnes-Hut instead saves an initial
frame in a binary file and clamps the save interval to at least one.

The simulation state CSVs use six decimal places, so a very small change may not
appear in the saved text. The ring GIFs show an x-y projection and use the 90th
percentile of radial position to expand the view, which keeps the main group
visible but can leave distant bodies outside the frame; playback speed is not
the simulation time step.

## Reading the performance results

For N bodies and S steps, a direct run evaluates this many ordered pairs,
including self-pairs and both directions of each interaction:

$$
P=N^2S
$$

This gives quadratic work per step, so doubling N means roughly four times as
many pairs, even when GPU parallelism reduces elapsed time. Simulation buffers
need linear memory, although keeping many animation frames adds storage that
also grows with the number of saved frames.

If the reported compute time is C milliseconds, the direct versions calculate:

$$
\mathrm{ms\ per\ step}=\frac{C}{S}
$$

$$
\mathrm{billion\ pairs/second}=\frac{P}{C\times10^6},\qquad
\mathrm{estimated\ GFLOPS}=\frac{20P}{C\times10^6}
$$

The estimate of 20 operations comes from three displacement subtractions, six
operations for squared distance and softening, one square root or reciprocal
square root, four operations for the acceleration scale, and six component
multiply/add operations. This treats division and square root as single
operations and excludes integration and other overhead, so it is not a count
of hardware instructions or an exact hardware FLOP measurement.

Barnes-Hut multiplies mass by G once before the pair calculation, reducing its
pair helper to an estimated 19 operations, but its total work also includes
sorting, tree building and traversal. Its N-squared throughput is explicitly an
**all-pairs-equivalent rate**, not a count of actual tree interactions or GFLOPS.

Each scaling CSV records the settings actually reported by the executable, and
the direct plots show compute time per step, total runtime and throughput, with
an N-squared reference on the compute chart. The compute timer covers different
work in different implementations, as explained in their READMEs; CPU process
time is also different from wall time, so end-to-end comparisons should use
compatible settings and timing boundaries, including the same saving policy.
Compilation and GIF creation are outside the executable's total runtime.

[compare.ipynb](./compare.ipynb) provides a separate comparison notebook, but
contains its own embedded sources rather than importing the folder notebooks,
so their later changes are not automatically reflected there. Match the initial
conditions, parameters, step counts and compiler options before treating runs as
equivalent, and remember that host-side checks cannot establish CUDA performance.
