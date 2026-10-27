import contextlib
import copy
import io
import json
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import benchmark


ROOT = Path(__file__).resolve().parents[1]


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("g++"):
            raise unittest.SkipTest("The CPU integration tests need g++.")
        cls.build = tempfile.TemporaryDirectory(prefix="nbody-test-build-")
        cls.addClassCleanup(cls.build.cleanup)
        cls.executable = Path(cls.build.name) / "cpu.out"
        subprocess.run(
            ["g++", "-O3", '-DPOSITIONS_FILE="positions.csv"', '-DFINAL_STATE_FILE="final_state.csv"',
             "-o", str(cls.executable), str(ROOT / "cpu/nbody_cpu.cpp")],
            text=True, check=True
        )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nbody-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings = json.loads((ROOT / "comparison_config.json").read_text())
        settings["main"] = {"bodies": 8, "steps": 6, "save_every": 2}
        settings["benchmark"] = {"body_counts": [2, 4, 8], "steps": 200, "save_every": 0}
        (self.root / "comparison_config.json").write_text(json.dumps(settings))
        self.settings = benchmark.load_settings(self.root)
        for specification in benchmark.VARIANTS.values():
            folder = self.root / specification["folder"]
            folder.mkdir()
            shutil.copyfile(self.executable, folder / specification["executable"])
            (folder / specification["executable"]).chmod(0o755)
        self.addCleanup(patch.stopall)
        patch.object(benchmark, "PROJECT_ROOT", self.root).start()
        patch.object(benchmark, "hardware_info", return_value={
            "host": "test fixture", "system": "test fixture", "cpu": "test fixture",
            "gpu": "test fixture", "cuda_visible_devices": "all",
        }).start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def run_cpu(self):
        main = benchmark.run_main("cpu", self.settings)
        benchmark.render_animation("cpu", self.settings)
        rows = benchmark.run_scaling("cpu", self.settings)
        return main, rows

    def make_all_exports(self):
        main, rows = self.run_cpu()
        cpu = self.root / "cpu"
        frames, masses, steps = benchmark.load_frames(cpu / "positions.csv", "cpu", self.settings["main"])
        for index, (variant, specification) in enumerate(benchmark.VARIANTS.items()):
            if variant == "cpu":
                continue
            folder = self.root / specification["folder"]
            if variant == "barnes_hut":
                data = np.empty((*frames.shape[:2], 4), dtype=np.float32)
                data[:, :, :3] = frames
                data[:, :, 3] = masses
                with (folder / "frames.bin").open("wb") as stream:
                    stream.write(struct.pack("ii", len(masses), len(steps)))
                    data.tofile(stream)
                state = pd.read_csv(cpu / "final_state.csv").rename(columns={"id": "i", "mass": "m"})
                state.to_csv(folder / "final_output.csv", index=False)
                artifacts = [folder / "frames.bin", folder / "final_output.csv"]
            else:
                for name in ["positions.csv", "final_state.csv"]:
                    shutil.copyfile(cpu / name, folder / name)
                artifacts = [folder / "positions.csv", folder / "final_state.csv"]
            shutil.copyfile(cpu / "nbody.gif", folder / "nbody.gif")
            benchmark.write_metadata(
                folder / "run.json", variant, self.settings, [*artifacts, folder / "nbody.gif"],
                metrics=main["metrics"], initial_state_id=main["initial_state_id"], frame_steps=steps.tolist()
            )
            data = pd.DataFrame(rows)
            data["version"] = specification["label"]
            data["compute time ms"] /= index + 1
            data["ms per step"] = data["compute time ms"] / data["steps"]
            data["total ms"] /= index + 1
            if variant == "barnes_hut":
                data["scenario"] = 2
                data["opening angle"] = self.settings["barnes_hut"]["opening_angle"]
            data.to_csv(folder / "scaling.csv", index=False)
            benchmark.write_metadata(
                folder / "scaling.json", variant, self.settings, [folder / "scaling.csv"],
                initial_state_id=main["initial_state_id"], timing_scope="Test fixture"
            )

    def execute_comparison(self):
        notebook = json.loads((ROOT / "compare.ipynb").read_text())
        namespace = {"__name__": "__main__"}
        previous = Path.cwd()

        def show():
            for number in plt.get_fignums():
                plt.figure(number).canvas.draw()
            plt.close("all")

        try:
            import os
            os.chdir(self.root)
            with patch("IPython.display.display"), patch("matplotlib.pyplot.show", side_effect=show), \
                patch("benchmark.run_experiment", side_effect=AssertionError("Comparison must not run a solver")), \
                warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"compare.ipynb:cell{index}", "exec"), namespace)
            return namespace, recorded
        finally:
            os.chdir(previous)
            plt.close("all")

    def test_shared_defaults(self):
        settings = benchmark.load_settings(ROOT)
        self.assertEqual(settings["main"], {"bodies": 10000, "steps": 1000, "save_every": 2})
        self.assertEqual(settings["benchmark"]["steps"], 100)
        self.assertEqual(settings["benchmark"]["save_every"], 0)
        self.assertEqual(settings["benchmark"]["body_counts"], [16, 64, 256, 1024, 2048, 4096, 8192, 10000])
        self.assertEqual(
            [settings[name] for name in ["seed", "time_step", "gravitational_constant", "softening"]],
            [42, 0.005, 0.001, 0.01]
        )
        self.assertEqual(settings["barnes_hut"]["scenario"], 2)

    def test_all_ring_initializers_match(self):
        paths = {
            "cpu": "cpu/cpu.ipynb", "naive": "gpu_naive/naive.ipynb",
            "tiled": "gpu_tiled/tiled.ipynb", "optimised": "gpu_optimised/optimised.ipynb",
            "tree": "gpu_barnes_hut/barnes_hut.ipynb",
        }

        def initializer(variant, name):
            source = "".join(json.loads((ROOT / paths[variant]).read_text())["cells"][3]["source"])
            match = re.search(r"\bvoid\s+" + name + r"\s*\(", source)
            self.assertIsNotNone(match)
            opening = source.index("{", match.start())
            end, depth = opening + 1, 1
            while depth:
                depth += (source[end] == "{") - (source[end] == "}")
                end += 1
            return source[match.start():end].replace(name + "(", "init_" + variant + "(", 1)

        header = """
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
struct Body { float x, y, z, mass; };
struct Velocity { float x, y, z; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
float3 make_float3(float x, float y, float z) { return {x, y, z}; }
float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
"""
        main = """
int main()
{
    for (int n : {16, 64, 10000})
    {
        std::vector<Body> cpu(n);
        std::vector<Velocity> cpuVelocity(n);
        std::vector<float4> gpu(n), treeVelocity(n);
        std::vector<float3> gpuVelocity(n);
        init_cpu(cpu.data(), cpuVelocity.data(), n);
        for (auto init : {init_naive, init_tiled, init_optimised})
        {
            init(gpu.data(), gpuVelocity.data(), n);
            for (int i = 0; i < n; i++)
            {
                if (memcmp(&cpu[i], &gpu[i], sizeof(Body)) ||
                    memcmp(&cpuVelocity[i], &gpuVelocity[i], sizeof(Velocity)))
                {
                    return 1;
                }
            }
        }
        init_tree(gpu.data(), treeVelocity.data(), n);
        for (int i = 0; i < n; i++)
        {
            if (memcmp(&cpu[i], &gpu[i], sizeof(Body)) ||
                memcmp(&cpuVelocity[i], &treeVelocity[i], sizeof(Velocity)))
            {
                return 2;
            }
        }
    }
}
"""
        definitions = [initializer(variant, "initRing" if variant == "tree" else "initBodies")
                       for variant in paths]
        source = self.root / "initializers.cpp"
        source.write_text(header + "\n".join(definitions) + main)
        executable = self.root / "initializers.out"
        for seed in [42, 123]:
            with self.subTest(seed=seed):
                subprocess.run(["g++", "-std=c++11", "-O3", f"-DINITIAL_SEED={seed}",
                                str(source), "-o", str(executable)], check=True)
                subprocess.run([str(executable)], check=True)

    def test_load_settings_reads_configuration(self):
        settings = copy.deepcopy(self.settings)
        settings["seed"] = 123
        settings["main"]["bodies"] = 16
        (self.root / "comparison_config.json").write_text(json.dumps(settings))
        self.assertEqual(benchmark.load_settings(self.root), settings)

    def test_cpu_frames_and_preserved_main_outputs(self):
        main = benchmark.run_main("cpu", self.settings)
        folder = self.root / "cpu"
        positions, masses, steps = benchmark.load_frames(folder / "positions.csv", "cpu", self.settings["main"])
        self.assertEqual(positions.shape, (4, 8, 3))
        np.testing.assert_array_equal(steps, [0, 2, 4, 6])
        self.assertEqual(len(masses), 8)
        gif = benchmark.render_animation("cpu", self.settings)
        with Image.open(gif) as image:
            self.assertEqual(image.n_frames, 4)
        before = {name: benchmark.file_digest(folder / name) for name in [
            "positions.csv", "final_state.csv", "nbody.gif"
        ]}
        rows = benchmark.run_scaling("cpu", self.settings)
        self.assertEqual([row["bodies"] for row in rows], [2, 4, 8])
        self.assertTrue(all(row["steps"] == 200 and row["save every"] == 0 for row in rows))
        self.assertEqual(before, {name: benchmark.file_digest(folder / name) for name in before})
        self.assertEqual(main["configuration_id"], benchmark.configuration_id(self.settings))
        metadata = benchmark.load_export(self.root, "cpu", "benchmark")
        self.assertEqual(metadata["initial_state_id"], main["initial_state_id"])
        self.assertEqual(metadata["artifacts"]["scaling.csv"], benchmark.file_digest(folder / "scaling.csv"))

    def test_load_export_reads_saved_metadata(self):
        metadata = benchmark.run_main("cpu", self.settings)
        with (self.root / "cpu/final_state.csv").open("a") as stream:
            stream.write("\n")
        self.assertEqual(benchmark.load_export(self.root, "cpu", "main"), metadata)

    def test_failed_main_does_not_replace_previous_outputs(self):
        benchmark.run_main("cpu", self.settings)
        path = self.root / "cpu/run.json"
        before = path.read_bytes()
        with patch("benchmark.run_experiment", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                benchmark.run_main("cpu", self.settings)
        self.assertEqual(path.read_bytes(), before)
        benchmark.load_export(self.root, "cpu", "main")

    def test_command_order_and_report_parsing(self):
        run = self.settings["main"]
        fields = {
            "bodies": run["bodies"], "steps": run["steps"], "save every": run["save_every"],
            "time step": 0.005, "gravitational constant": 0.001, "softening": 0.01,
            "random seed": 42, "compute time ms": 2, "total time ms": 3,
            "setup time ms": 0.5, "saving time ms": 0.5, "scenario": 2,
            "opening angle": 0.6, "resort every": 4, "check bodies": 0, "max frame bodies": 32768,
        }
        text = "\n".join(f"{name}: {value}" for name, value in fields.items()) + "\nmode: tree\n"
        result = SimpleNamespace(returncode=0, stdout=text, stderr="")
        for variant in benchmark.VARIANTS:
            with self.subTest(variant=variant), patch("benchmark.subprocess.run", return_value=result) as runner:
                report = benchmark.run_experiment(variant, self.settings, run, self.root)
                self.assertEqual(report["ms per step"], 2 / run["steps"])
                self.assertEqual(report["total ms"], 3)
                self.assertIsInstance(report["bodies"], int)
                self.assertTrue(runner.call_args.kwargs["check"])
                args = runner.call_args.args[0][1:]
                if variant == "barnes_hut":
                    self.assertEqual(args[:6], ["8", "6", "0.005", "0.001", "0.01", "2"])
                else:
                    self.assertEqual(args[:6], ["8", "6", "0.005", "2", "0.001", "0.01"])
                if variant == "optimised":
                    self.assertEqual(args[-1], str(self.settings["optimised_slices"]))

    def test_solver_failure_propagates(self):
        executable = self.root / "failure.sh"
        executable.write_text("#!/bin/sh\nexit 1\n")
        executable.chmod(0o755)
        with self.assertRaises(subprocess.CalledProcessError):
            benchmark.run_experiment("cpu", self.settings, self.settings["main"], self.root,
                                     executable=executable)

    def test_empty_and_cpu_only_comparisons(self):
        namespace, messages = self.execute_comparison()
        self.assertTrue(namespace["results"].empty)
        self.assertTrue(messages)
        self.run_cpu()
        namespace, messages = self.execute_comparison()
        self.assertEqual(len(namespace["results"]), 3)
        self.assertEqual(list(namespace["final_states"]), ["cpu"])
        self.assertNotIn("compute_speedup", namespace)

    def test_all_exports_and_hardware_mismatch(self):
        self.make_all_exports()
        namespace, messages = self.execute_comparison()
        self.assertEqual(len(namespace["results"]), 15)
        self.assertEqual(namespace["compute_speedup"].shape, (3, 4))
        self.assertEqual(len(namespace["cards"]), 5)
        self.assertEqual(len(namespace["final_states"]), 5)
        self.assertEqual(len(namespace["differences"]), 4)
        self.assertTrue(namespace["speedups_allowed"])
        path = self.root / "gpu_naive/scaling.json"
        metadata = json.loads(path.read_text())
        metadata["hardware"]["host"] = "another test machine"
        path.write_text(json.dumps(metadata))
        namespace, messages = self.execute_comparison()
        self.assertFalse(namespace["speedups_allowed"])
        self.assertNotIn("compute_speedup", namespace)
        self.assertTrue(any("Hardware or starting bodies differ" in str(message.message) for message in messages))

    def test_mismatched_build_and_missing_cpu_exports(self):
        self.make_all_exports()
        path = self.root / "gpu_naive/scaling.json"
        metadata = json.loads(path.read_text())
        metadata["executable_sha256"] = "another build"
        path.write_text(json.dumps(metadata))
        namespace, messages = self.execute_comparison()
        self.assertNotIn("naive", namespace["benchmark_metadata"])
        self.assertTrue(any("different builds" in str(message.message) for message in messages))
        for name in ["run.json", "scaling.json"]:
            (self.root / "cpu" / name).unlink()
        namespace, messages = self.execute_comparison()
        self.assertNotIn("cpu", namespace["benchmark_metadata"])
        self.assertNotIn("compute_speedup", namespace)

    def test_comparison_omits_different_settings(self):
        self.run_cpu()
        path = self.root / "cpu/scaling.json"
        metadata = json.loads(path.read_text())
        metadata["configuration"]["time_step"] *= 2
        path.write_text(json.dumps(metadata))
        namespace, messages = self.execute_comparison()
        self.assertNotIn("cpu", namespace["benchmark_metadata"])
        self.assertIn("cpu", namespace["main_exports"])
        self.assertTrue(namespace["results"].empty)
        self.assertTrue(any("Different settings" in str(message.message) for message in messages))

    def test_bad_binary_frames(self):
        path = self.root / "gpu_barnes_hut/frames.bin"
        path.write_bytes(struct.pack("ii", 8, 4) + b"\0" * 16)
        with self.assertRaisesRegex(ValueError, "reshape"):
            benchmark.load_frames(path, "barnes_hut", self.settings["main"])


if __name__ == "__main__":
    unittest.main()
