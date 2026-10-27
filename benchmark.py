"""Shared run settings and result exports for the simulation notebooks."""

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "cpu": {"folder": "cpu", "executable": "nbody_cpu.out", "label": "CPU"},
    "naive": {"folder": "gpu_naive", "executable": "nbody_naive.out", "label": "Naive CUDA"},
    "tiled": {"folder": "gpu_tiled", "executable": "nbody_tiled.out", "label": "Tiled CUDA"},
    "optimised": {"folder": "gpu_optimised", "executable": "nbody_opt.out", "label": "Optimised CUDA"},
    "barnes_hut": {"folder": "gpu_barnes_hut", "executable": "nbody_tree.out", "label": "Barnes-Hut CUDA"},
}
SCHEMA_VERSION = 1


def configuration_id(settings):
    encoded = json.dumps(settings, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_settings(root=PROJECT_ROOT):
    return json.loads((Path(root) / "comparison_config.json").read_text())


def hardware_info():
    hardware = {
        "host": platform.node(),
        "system": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        hardware["gpu"] = result.stdout.strip()
    return hardware


def parse_report(text):
    pattern = r"^([^:\n]+):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[ \t]*$"
    return {name.strip(): float(value) for name, value in re.findall(pattern, text, re.MULTILINE)}


def run_experiment(variant, settings, run, directory, executable=None, tree_overrides=None):
    specification = VARIANTS[variant]
    executable = Path(executable or PROJECT_ROOT / specification["folder"] / specification["executable"]).resolve()
    tree = settings["barnes_hut"] | (tree_overrides or {})
    physics = [settings["time_step"], settings["gravitational_constant"], settings["softening"]]
    arguments = [run["bodies"], run["steps"], physics[0]]
    if variant == "barnes_hut":
        arguments += [
            *physics[1:], run["save_every"], tree["scenario"], tree["opening_angle"], "tree",
            tree["resort_every"], tree["max_frame_bodies"], tree["check_bodies"]
        ]
    else:
        arguments += [run["save_every"], *physics[1:]]
        if variant == "optimised":
            arguments.append(settings["optimised_slices"])
    command = [str(executable), *map(str, arguments)]
    result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE, text=True, check=True)
    report = parse_report(result.stdout)
    integer_fields = {
        "bodies", "steps", "save every", "random seed", "block size", "slices per body",
        "scenario", "resort every", "leaf size", "tree leaves", "tree nodes", "max frame bodies",
        "check bodies", "checked bodies",
    }
    for name in integer_fields.intersection(report):
        report[name] = int(report[name])
    report["ms per step"] = report["compute time ms"] / run["steps"]
    report["total ms"] = report.pop("total time ms")
    return report


def write_metadata(path, variant, settings, artifacts, **details):
    specification = VARIANTS[variant]
    executable = PROJECT_ROOT / specification["folder"] / specification["executable"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "configuration": settings,
        "configuration_id": configuration_id(settings),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_info(),
        "executable_sha256": file_digest(executable),
        "artifacts": {Path(name).name: file_digest(name) for name in artifacts},
        **details,
    }
    Path(path).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def load_export(root, variant, kind):
    folder = Path(root) / VARIANTS[variant]["folder"]
    path = folder / ("run.json" if kind == "main" else "scaling.json")
    return json.loads(path.read_text())


def run_main(variant, settings):
    folder = PROJECT_ROOT / VARIANTS[variant]["folder"]
    frames_name = "frames.bin" if variant == "barnes_hut" else "positions.csv"
    final_name = "final_output.csv" if variant == "barnes_hut" else "final_state.csv"
    # Keep the previous outputs if the solver or frame loading fails.
    with TemporaryDirectory(prefix=".nbody-main-", dir=folder) as directory:
        report = run_experiment(variant, settings, settings["main"], directory)
        frames, masses, steps = load_frames(Path(directory) / frames_name, variant, settings["main"])
        canonical = "\n".join(
            ",".join(f"{value:.6f}" for value in [*point, mass])
            for point, mass in zip(frames[0], masses)
        )
        (folder / "run.json").unlink(missing_ok=True)
        for name in [frames_name, final_name]:
            (Path(directory) / name).replace(folder / name)
    initial_id = hashlib.sha256(canonical.encode()).hexdigest()
    metadata = write_metadata(
        folder / "run.json", variant, settings, [folder / frames_name, folder / final_name],
        metrics=report, initial_state_id=initial_id, frame_steps=steps.tolist()
    )
    print(f"Saved {folder.name}/run.json for {settings['main']['bodies']:,} bodies and {settings['main']['steps']:,} steps.")
    return metadata


def run_scaling(variant, settings):
    folder = PROJECT_ROOT / VARIANTS[variant]["folder"]
    main = load_export(PROJECT_ROOT, variant, "main")
    (folder / "scaling.json").unlink(missing_ok=True)
    rows = []
    for bodies in settings["benchmark"]["body_counts"]:
        run = {"bodies": bodies, "steps": settings["benchmark"]["steps"],
               "save_every": settings["benchmark"]["save_every"]}
        print(f"Running {VARIANTS[variant]['label']} with {bodies:,} bodies")
        with TemporaryDirectory(prefix="nbody-benchmark-") as directory:
            report = run_experiment(variant, settings, run, directory)
        rows.append({"version": VARIANTS[variant]["label"], **report})
    path = folder / "scaling.csv"
    columns = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_metadata(
        folder / "scaling.json", variant, settings, [path],
        initial_state_id=main["initial_state_id"],
        timing_scope="Host compute wall time" if variant in ["cpu", "barnes_hut"] else "CUDA events",
    )
    return rows


def load_frames(path, variant, run):
    import numpy as np
    import pandas as pd

    expected_steps = np.arange(0, run["steps"] + 1, run["save_every"])
    if variant == "barnes_hut":
        with Path(path).open("rb") as stream:
            bodies, count = struct.unpack("ii", stream.read(8))
            raw = np.fromfile(stream, dtype=np.float32)
        data = raw.reshape(count, bodies, 4)
        positions, masses = data[:, :, :3], data[0, :, 3]
    else:
        data = pd.read_csv(path)
        data = data.sort_values(["step", "id"])
        positions = data[["x", "y", "z"]].to_numpy().reshape(-1, run["bodies"], 3)
        masses = data["mass"].to_numpy().reshape(-1, run["bodies"])[0]
    return positions, masses, expected_steps


def render_animation(variant, settings, filename="nbody.gif"):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    folder = PROJECT_ROOT / VARIANTS[variant]["folder"]
    metadata = load_export(PROJECT_ROOT, variant, "main")
    frames_name = "frames.bin" if variant == "barnes_hut" else "positions.csv"
    positions, masses, steps = load_frames(folder / frames_name, variant, settings["main"])
    radii = np.percentile(np.hypot(positions[:, :, 0], positions[:, :, 1]), 90, axis=1)
    view_size = np.maximum.accumulate(np.maximum(1.3 * radii, 0.1))
    fig, axis = plt.subplots(figsize=(6, 6), dpi=80)
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    dots = axis.scatter(positions[0, :, 0], positions[0, :, 1],
        s=1.5 * masses, c=masses, cmap="viridis", alpha=0.8, linewidths=0)

    def draw(index):
        dots.set_offsets(positions[index, :, :2])
        axis.set_xlim(-view_size[index], view_size[index])
        axis.set_ylim(-view_size[index], view_size[index])
        axis.set_title(f"{VARIANTS[variant]['label']} | step {steps[index]} | t = {steps[index] * settings['time_step']:.3f}")
        return dots,

    path = folder / filename
    try:
        FuncAnimation(fig, draw, frames=len(steps)).save(path, writer=PillowWriter(fps=15))
    finally:
        plt.close(fig)
    metadata["artifacts"][filename] = file_digest(path)
    (folder / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved {folder.name}/{filename} with {len(steps)} frames.")
    return path
