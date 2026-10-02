"""Tests for the timing comparison in demos/benchmark.py."""

import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

DEMOS = Path(__file__).resolve().parents[1] / "demos"
SCRIPT = DEMOS / "benchmark.py"
RESULTS = DEMOS / "benchmark_results.json"

# Set in the CI job that installs astroscrappy, lacosmic and matplotlib, so that the
# tests that need them cannot be skipped there by accident.
_REQUIRE = os.environ.get("DFCOSMIC_REQUIRE_BENCHMARK", "").lower() in {
    "1",
    "true",
    "yes",
}

CPU_CONFIGS = [
    "dfcosmic_torch",
    "dfcosmic_cpp",
    "astroscrappy_sepmed_false",
    "astroscrappy_sepmed_true",
    "lacosmic",
]

# The parameters of demos/HST.ipynb. Unlike the benchmark parameters (sigfrac=1), they
# give different values for the two lacosmic thresholds, so a swap would show.
HST_PARAMS = {
    "sigclip": 4.5,
    "sigfrac": 0.3,
    "objlim": 4.0,
    "gain": 7.0,
    "readnoise": 5.0,
}


def _need(module: str):
    if _REQUIRE:
        return importlib.import_module(module)
    return pytest.importorskip(module)


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestParameterMapping:
    """Every code must be called with the same L.A.Cosmic parameters."""

    def test_lacosmic(self, bench):
        kwargs = bench.call_kwargs("lacosmic", HST_PARAMS, niter=4, threads=1)
        assert kwargs == {
            "contrast": 4.0,
            "cr_threshold": 4.5,
            "neighbor_threshold": pytest.approx(1.35),
            "effective_gain": 7.0,
            "readnoise": 5.0,
            "maxiter": 4,
        }

    @pytest.mark.parametrize(
        "config, sepmed",
        [("astroscrappy_sepmed_false", False), ("astroscrappy_sepmed_true", True)],
    )
    def test_astroscrappy(self, bench, config, sepmed):
        kwargs = bench.call_kwargs(config, HST_PARAMS, niter=4, threads=2)
        assert kwargs == {**HST_PARAMS, "niter": 4, "sepmed": sepmed}

    @pytest.mark.parametrize(
        "config, use_cpp", [("dfcosmic_torch", False), ("dfcosmic_cpp", True)]
    )
    def test_dfcosmic_cpu(self, bench, config, use_cpp):
        kwargs = bench.call_kwargs(config, HST_PARAMS, niter=4, threads=2)
        assert kwargs == {
            **HST_PARAMS,
            "niter": 4,
            "device": "cpu",
            "cpu_threads": 2,
            "use_cpp": use_cpp,
        }

    def test_dfcosmic_gpu(self, bench):
        kwargs = bench.call_kwargs("dfcosmic_gpu", HST_PARAMS, niter=4, threads=2)
        assert kwargs == {**HST_PARAMS, "niter": 4, "device": "cuda"}

    @pytest.mark.parametrize("niter", [1, 4])
    def test_same_number_of_iterations_for_every_code(self, bench, niter):
        for config in bench.CONFIGS:
            kwargs = bench.call_kwargs(config, bench.PARAMS, niter=niter, threads=1)
            assert kwargs.get("niter", kwargs.get("maxiter")) == niter

    def test_kwargs_match_the_dfcosmic_signature(self, bench):
        from dfcosmic import lacosmic

        for config in ("dfcosmic_torch", "dfcosmic_cpp", "dfcosmic_gpu"):
            kwargs = bench.call_kwargs(config, bench.PARAMS, niter=1, threads=1)
            inspect.signature(lacosmic).bind(None, **kwargs)

    def test_kwargs_match_the_lacosmic_signature(self, bench):
        remove_cosmics = _need("lacosmic").remove_cosmics
        kwargs = bench.call_kwargs("lacosmic", bench.PARAMS, niter=1, threads=1)
        inspect.signature(remove_cosmics).bind(None, **kwargs)

    def test_unknown_configuration(self, bench):
        with pytest.raises(KeyError):
            bench.call_kwargs("unknown", bench.PARAMS, niter=1, threads=1)


class TestFakeData:
    def test_matches_the_astroscrappy_test_data(self, bench):
        fake_data = _need("astroscrappy.tests.fake_data")
        expected_image, expected_mask = fake_data.make_fake_data()
        image, mask = bench.make_fake_data((1001, 1001))
        np.testing.assert_array_equal(image, expected_image)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_larger_frame(self, bench):
        image, mask = bench.make_fake_data((1100, 1200))
        assert image.shape == mask.shape == (1100, 1200)
        assert image.dtype == np.float32
        assert 90 <= mask.sum() <= 100
        assert not mask[995:, :].any() and not mask[:, 995:].any()

    def test_too_small_frame_is_rejected(self, bench, tmp_path):
        with pytest.raises(ValueError, match="1001"):
            bench.frame_file((512, 512), tmp_path)

    def test_crowded_image_is_the_tiled_hst_frame(self, bench, tmp_path):
        fits = pytest.importorskip("astropy.io.fits")
        data = fits.getdata(DEMOS / "hst_im_ext3.fits").astype(np.float32)
        path = bench.frame_file((1000, 900), tmp_path, "hst")
        with np.load(path) as frame:
            image, injected = frame["image"], frame["injected"]
        assert image.shape == (1000, 900) and image.dtype == np.float32
        assert not injected.any()
        rows, columns = data.shape
        np.testing.assert_array_equal(image[:rows, :columns], data)
        np.testing.assert_array_equal(
            image[rows:, columns:], data[: 1000 - rows, : 900 - columns]
        )


class TestThreads:
    def test_threads_for(self, bench):
        assert bench.threads_for("dfcosmic_cpp", [16, 1, 4]) == [1, 4, 16]
        # The GPU and the single-threaded lacosmic are only timed at the two ends
        assert bench.threads_for("dfcosmic_gpu", [16, 1, 4]) == [1, 16]
        assert bench.threads_for("lacosmic", [2]) == [2]

    def test_child_environment(self, bench, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "7")
        monkeypatch.setenv("OMP_PROC_BIND", "true")
        monkeypatch.setenv("DFCOSMIC_DISABLE_CPP", "1")
        monkeypatch.setenv("SOMETHING_ELSE", "kept")
        env = bench.child_env(4)
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "4"
        assert env["OPENBLAS_NUM_THREADS"] == "4"
        assert "OMP_PROC_BIND" not in env
        assert "DFCOSMIC_DISABLE_CPP" not in env
        assert env["SOMETHING_ELSE"] == "kept"


def test_run_and_report(bench, tmp_path):
    """Run every available CPU configuration on a small frame, then report."""
    available = bench.probe()["available"]
    configs = [config for config in CPU_CONFIGS if available[config]]
    assert "dfcosmic_torch" in configs
    if _REQUIRE:
        assert configs == CPU_CONFIGS

    output = tmp_path / "results.json"
    command = [sys.executable, str(SCRIPT), "run", "--quick", "--threads", "2"]
    command += ["--configs", *configs, "--output", str(output)]
    command += ["--frame-dir", str(tmp_path), "--max-background", "1000"]
    subprocess.run(command, check=True)

    experiment = json.loads(output.read_text())["experiments"]["niter1"]
    assert experiment["settings"]["shape"] == [1001, 1001]
    assert experiment["versions"]["dfcosmic"] is not None
    assert experiment["system"]["cpu"]
    runs = {run["config"]: run for run in experiment["runs"]}
    assert sorted(runs) == sorted(configs)
    for config, run in runs.items():
        assert "error" not in run, run
        assert run["threads"] == 2
        assert run["limit_ok"] and run["input_unchanged"]
        assert len(run["times_s"]) == len(run["cpu_s"]) == 1
        assert run["passes"] == 1
        # All codes find the 100 injected cosmic rays with these parameters
        assert run["n_injected_flagged"] == run["n_injected"] == 100
        assert run["kwargs"] == bench.call_kwargs(config, bench.PARAMS, 1, 2)
        if config != "lacosmic":  # lacosmic does not use threads
            assert max(run["cpu_s"]) / min(run["times_s"]) < 2.3

    assert bench.find_problems(experiment) == []
    for table in (bench.timing_table, bench.ratio_table, bench.checks_table):
        text = table(experiment)
        for config in configs:
            assert bench.CONFIGS[config]["label"] in text
    assert "1001 × 1001" in bench.provenance(experiment)
    assert "neighbor_threshold=6.0" in bench.calls_table(experiment) or (
        "lacosmic" not in configs
    )

    report = [sys.executable, str(SCRIPT), "report", "--input", str(output)]
    done = subprocess.run(
        [*report, "--no-figures"], check=True, capture_output=True, text=True
    )
    assert "No warnings." in done.stdout

    _need("matplotlib")
    subprocess.run([*report, "--outdir", str(tmp_path)], check=True)
    for name in ("comparison.png", "comparison_dark.png"):
        assert (tmp_path / name).stat().st_size > 10_000


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text())


class TestCommittedResults:
    """The results file that the README, the docs and the paper quote."""

    def test_main_experiment(self, bench, results):
        experiment = results["experiments"]["niter1"]
        settings = experiment["settings"]
        assert settings["shape"] == list(bench.FRAME_SHAPE)
        assert settings["params"] == bench.PARAMS
        assert settings["threads"] == list(bench.THREADS)
        assert settings["niter"] == 1

        summary = bench.summarise(experiment)
        expected = {
            (config, threads)
            for config in bench.CONFIGS
            for threads in bench.threads_for(config, bench.THREADS)
        }
        assert set(summary) == expected

    @pytest.mark.parametrize("key", ["niter1", "niter4", "hst_niter1"])
    def test_every_point_is_complete_and_matched(self, bench, results, key):
        experiment = results["experiments"][key]
        settings = experiment["settings"]
        assert settings["params"] == bench.IMAGES[settings["image"]]["params"]
        calls = settings["rounds"] * settings["repeats"]
        assert calls >= 6
        assert not [run for run in experiment["runs"] if "error" in run]
        for run in experiment["runs"]:
            assert run["limit_ok"] and run["input_unchanged"]
            assert run["kwargs"] == bench.call_kwargs(
                run["config"], settings["params"], settings["niter"], run["threads"]
            )
        for stats in bench.summarise(experiment).values():
            assert stats["n"] == calls
        assert bench.find_problems(experiment) == []

    def test_results_for_the_previous_version(self, bench, results):
        """The file the before-and-after comparison is made with."""
        before = json.loads((DEMOS / "benchmark_results_v0.1.0.json").read_text())
        assert before["experiments"]["niter1"]["versions"]["dfcosmic"] == "0.1.0"
        old, new = before["experiments"]["niter1"], results["experiments"]["niter1"]
        for key in ("image", "shape", "params", "niter", "threads"):
            assert old["settings"][key] == new["settings"][key]
        assert "| 16 |" in bench.speedup_table(old, new, ["dfcosmic_cpp"])
