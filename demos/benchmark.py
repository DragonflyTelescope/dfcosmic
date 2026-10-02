"""
Like-for-like timing comparison of dfcosmic with astroscrappy and lacosmic.

Usage, from the repository root::

    python demos/benchmark.py run                          # niter=1, 1 to 16 threads
    python demos/benchmark.py run --niter 4 --threads 1 2  # the same codes at niter=4
    python demos/benchmark.py report                       # tables and figures
    python demos/benchmark.py run --quick                  # 1001 x 1001, to check the set-up

The results are stored in ``demos/benchmark_results.json`` together with the hardware,
the package versions and every raw timing. ``demos/Comparison.ipynb`` displays them.

How the comparison is kept like-for-like
----------------------------------------
* Every code gets the same image and the same L.A.Cosmic parameters. ``call_kwargs``
  is the only place where the parameter names of the three packages are mapped.
* Every code runs the same number of iterations (``--niter``).
* Each (configuration, number of threads) is timed in a fresh Python process that only
  imports the code under test. The process makes one untimed warm-up call on the full
  frame, followed by ``--repeats`` timed calls. This is done ``--rounds`` times, in a
  different random order each time, so every point is the median of
  ``rounds * repeats`` calls from independent processes (15 by default).
* The number of threads is set through the environment of the process
  (``OMP_NUM_THREADS`` etc.), through ``threadpoolctl`` and, for dfcosmic, through
  ``cpu_threads``. Each process records the size of its thread pools and its CPU time,
  so that the limit can be checked afterwards.
* Timings are end to end: a numpy array goes in, and a cleaned numpy array and a mask
  come out. For the GPU this includes the transfer to and from the device.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "benchmark_results.json"

# The test images, each with its L.A.Cosmic parameters, named as in the original IRAF
# script (and in dfcosmic and astroscrappy).
IMAGES = {
    # The image the comparison has always used: very few cosmic rays
    "synthetic": {
        "description": "synthetic image from the astroscrappy test suite, with 100 "
        "stars and 100 cosmic rays",
        "params": {
            "sigclip": 6.0,
            "sigfrac": 1.0,
            "objlim": 2.0,
            "gain": 1.0,
            "readnoise": 10.0,
        },
    },
    # A crowded image: about 3% of its pixels are cosmic rays
    "hst": {
        "description": "HST WFPC2 image of MS 1137+67 from van Dokkum 2001 "
        "(demos/hst_im_ext3.fits), tiled to the frame size",
        "params": {
            "sigclip": 4.5,
            "sigfrac": 0.3,
            "objlim": 4.0,
            "gain": 7.0,
            "readnoise": 5.0,
        },
    },
}
PARAMS = IMAGES["synthetic"]["params"]
FRAME_SHAPE = (4000, 6500)
SEED = 200
THREADS = (1, 2, 4, 8, 16)

# "threads": "all" is timed at every thread count. "ends" is timed at the smallest and
# the largest only, because the number of CPU threads does not matter for it.
CONFIGS = {
    "dfcosmic_torch": {
        "package": "dfcosmic",
        "label": "dfcosmic · CPU, PyTorch only",
        "threads": "all",
        "note": "What `pip install dfcosmic` gives.",
    },
    "dfcosmic_cpp": {
        "package": "dfcosmic",
        "label": "dfcosmic · CPU, C++ median filter",
        "threads": "all",
        "note": "Needs the optional C++ extension, built from source.",
    },
    "dfcosmic_gpu": {
        "package": "dfcosmic",
        "label": "dfcosmic · GPU",
        "threads": "ends",
        "note": "Includes the transfer of the image to and from the GPU.",
    },
    "astroscrappy_sepmed_false": {
        "package": "astroscrappy",
        "label": "astroscrappy · true median (sepmed=False)",
        "threads": "all",
        "note": "Same median filter as the original algorithm.",
    },
    "astroscrappy_sepmed_true": {
        "package": "astroscrappy",
        "label": "astroscrappy · separable median (sepmed=True)",
        "threads": "all",
        "note": "astroscrappy's default. A different, faster algorithm.",
    },
    "lacosmic": {
        "package": "lacosmic",
        "label": "lacosmic (single-threaded)",
        "threads": "ends",
        "note": "numpy and scipy only, so it does not use more than one thread.",
    },
}

_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
# A job measured while other processes kept more than this many CPUs busy is repeated
MAX_BACKGROUND_CPUS = 1.0
# Inherited settings that would change the behaviour of a code under test
_SCRUB_ENV_PREFIXES = (
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "KMP_",
    "GOMP_",
    "NUMEXPR_",
    "VECLIB_",
    "DFCOSMIC_",
)


# --------------------------------------------------------------------------------------
# Test image
# --------------------------------------------------------------------------------------
# gaussian() and make_fake_data() are taken from the astroscrappy test suite
# (astroscrappy/tests/fake_data.py, BSD 3-clause). The only change is that the image
# size is a parameter. As in the original, the 100 sources and the 100 cosmic rays are
# placed in the first 1001 x 1001 pixels whatever the size of the image.
def gaussian(image_shape, x0, y0, brightness, fwhm):
    x = np.arange(image_shape[1])
    y = np.arange(image_shape[0])
    x2d, y2d = np.meshgrid(x, y)

    sig = fwhm / 2.35482

    normfactor = brightness / 2.0 / np.pi * sig**-2.0
    exponent = -0.5 * sig**-2.0
    exponent *= (x2d - x0) ** 2.0 + (y2d - y0) ** 2.0

    return normfactor * np.exp(exponent)


def make_fake_data(size=(1001, 1001)):
    """
    Generate fake data that can be used to test the detection and cleaning algorithms

    Returns
    -------
    imdata : numpy float array
        Fake Image data
    crmask : numpy boolean array
        Boolean mask of locations of injected cosmic rays
    """
    # Set a seed so that the tests are repeatable
    np.random.seed(SEED)

    # Create a simulated image to use in our tests
    imdata = np.zeros(size, dtype=np.float32)

    # Add sky and sky noise
    imdata += 200

    psf_sigma = 3.5

    # Add some fake sources
    for i in range(100):
        x = np.random.uniform(low=0.0, high=1001)
        y = np.random.uniform(low=0.0, high=1001)
        brightness = np.random.uniform(low=1000.0, high=30000.0)
        imdata += gaussian(imdata.shape, x, y, brightness, psf_sigma)

    # Add the poisson noise
    imdata = np.float32(np.random.poisson(imdata))

    # Add readnoise
    imdata += np.random.normal(0.0, 10.0, size=size)

    # Add 100 fake cosmic rays
    cr_x = np.random.randint(low=5, high=995, size=100)
    cr_y = np.random.randint(low=5, high=995, size=100)

    cr_brightnesses = np.random.uniform(low=1000.0, high=30000.0, size=100)

    imdata[cr_y, cr_x] += cr_brightnesses
    imdata = imdata.astype("f4")

    # Make a mask where the detected cosmic rays should be
    crmask = np.zeros(size, dtype=bool)
    crmask[cr_y, cr_x] = True
    return imdata, crmask


def make_hst_data(size):
    """The HST image of the HST example, repeated to fill an image of the given size."""
    from astropy.io import fits

    data = fits.getdata(HERE / "hst_im_ext3.fits").astype(np.float32)
    repeats = (-(-size[0] // data.shape[0]), -(-size[1] // data.shape[1]))
    imdata = np.ascontiguousarray(np.tile(data, repeats)[: size[0], : size[1]])
    # No cosmic rays are injected: the real ones are not known individually
    return imdata, np.zeros(size, dtype=bool)


def frame_file(shape, frame_dir: Path, image_name: str = "synthetic") -> Path:
    """Generate the test image once and keep it on disk for the worker processes."""
    if image_name == "synthetic" and min(shape) < 1001:
        raise ValueError("the test image must be at least 1001 x 1001 pixels")
    frame_dir.mkdir(parents=True, exist_ok=True)
    name = f"seed{SEED}" if image_name == "synthetic" else image_name
    path = frame_dir / f"frame_{shape[0]}x{shape[1]}_{name}.npz"
    if not path.exists():
        print(f"Generating the {shape[0]} x {shape[1]} test image ...", flush=True)
        make = make_fake_data if image_name == "synthetic" else make_hst_data
        image, injected = make(tuple(shape))
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, image=image, injected=injected)
        os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------------------
# The codes under test
# --------------------------------------------------------------------------------------
def call_kwargs(config: str, params: dict, niter: int, threads: int) -> dict:
    """
    Keyword arguments with which a configuration is called.

    This is the only place where the L.A.Cosmic parameters are translated into the
    parameter names of each package.
    """
    package = CONFIGS[config]["package"]
    if package == "dfcosmic":
        kwargs = {
            "sigclip": params["sigclip"],
            "sigfrac": params["sigfrac"],
            "objlim": params["objlim"],
            "gain": params["gain"],
            "readnoise": params["readnoise"],
            "niter": niter,
        }
        if config == "dfcosmic_gpu":
            kwargs["device"] = "cuda"
        else:
            kwargs["device"] = "cpu"
            kwargs["cpu_threads"] = threads
            kwargs["use_cpp"] = config == "dfcosmic_cpp"
        return kwargs
    if package == "astroscrappy":
        # Everything else is left at astroscrappy's defaults
        return {
            "sigclip": params["sigclip"],
            "sigfrac": params["sigfrac"],
            "objlim": params["objlim"],
            "gain": params["gain"],
            "readnoise": params["readnoise"],
            "niter": niter,
            "sepmed": config == "astroscrappy_sepmed_true",
        }
    if package == "lacosmic":
        # lacosmic takes the threshold for neighbouring pixels directly, where the
        # other codes take it as a fraction (sigfrac) of the detection threshold.
        return {
            "contrast": params["objlim"],
            "cr_threshold": params["sigclip"],
            "neighbor_threshold": params["sigclip"] * params["sigfrac"],
            "effective_gain": params["gain"],
            "readnoise": params["readnoise"],
            "maxiter": niter,
        }
    raise ValueError(f"unknown configuration {config!r}")


def load_runner(config: str):
    """Import the code under test and return ``run(image, **kwargs) -> (clean, mask)``."""
    package = CONFIGS[config]["package"]
    if package == "dfcosmic":
        import torch
        from dfcosmic import lacosmic
        from dfcosmic.utils import cpp_median_available

        if config == "dfcosmic_cpp" and not cpp_median_available():
            raise RuntimeError("the dfcosmic C++ median filter has not been built")
        if config == "dfcosmic_gpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return lacosmic
    if package == "astroscrappy":
        from astroscrappy import detect_cosmics

        def run_astroscrappy(image, **kwargs):
            mask, clean = detect_cosmics(image, **kwargs)
            return clean, mask

        return run_astroscrappy
    if package == "lacosmic":
        from lacosmic import remove_cosmics

        return remove_cosmics
    raise ValueError(f"unknown configuration {config!r}")


def _call_counting_passes(config: str, run, image, kwargs: dict):
    """Call a code with its progress output on, and count the passes it executed."""
    if CONFIGS[config]["package"] == "lacosmic":
        from astropy import log

        with log.log_to_list() as records:
            clean, mask = run(image, **kwargs)
        text = "\n".join(record.getMessage() for record in records)
    else:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            clean, mask = run(image, **kwargs, verbose=True)
        text = buffer.getvalue()
    # All three codes announce every pass with a line containing "Iteration <n>"
    return clean, mask, text.count("Iteration")


# --------------------------------------------------------------------------------------
# Worker: one fresh process per (configuration, number of threads)
# --------------------------------------------------------------------------------------
def worker_main(args) -> None:
    config, threads = args.config, args.threads

    with np.load(args.frame) as frame:
        image, injected = frame["image"], frame["injected"]
    checksum = float(image.sum(dtype=np.float64))

    start = time.perf_counter()
    run = load_runner(config)
    import_s = time.perf_counter() - start
    kwargs = call_kwargs(config, IMAGES[args.image]["params"], args.niter, threads)

    from threadpoolctl import threadpool_info, threadpool_limits

    torch = sys.modules.get("torch")
    on_gpu = config == "dfcosmic_gpu"
    if torch is not None:
        torch.set_num_threads(threads)

    with threadpool_limits(limits=threads):
        pools = [
            {
                "user_api": pool["user_api"],
                "internal_api": pool["internal_api"],
                "num_threads": pool["num_threads"],
                "library": Path(pool["filepath"]).name,
            }
            for pool in threadpool_info()
        ]
        limit_ok = all(
            pool["num_threads"] == threads
            for pool in pools
            if pool["user_api"] == "openmp"
        )
        if torch is not None:
            limit_ok = limit_ok and torch.get_num_threads() == threads

        # Warm-up on the full frame. Not part of the statistic, but recorded.
        start = time.perf_counter()
        clean, mask, passes = _call_counting_passes(config, run, image, kwargs)
        first_call_s = time.perf_counter() - start

        times, cpu_times = [], []
        for _ in range(args.repeats):
            if on_gpu:
                torch.cuda.synchronize()
            cpu_start, start = time.process_time(), time.perf_counter()
            clean, mask = run(image, **kwargs)
            if on_gpu:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            cpu_times.append(time.process_time() - cpu_start)

    mask = np.asarray(mask, dtype=bool)
    record = {
        "config": config,
        "threads": threads,
        "kwargs": kwargs,
        "import_s": import_s,
        "first_call_s": first_call_s,
        "times_s": times,
        "cpu_s": cpu_times,
        "passes": passes,
        "n_flagged": int(mask.sum()),
        "n_injected": int(injected.sum()),
        "n_injected_flagged": int((mask & injected).sum()),
        "output_dtype": str(np.asarray(clean).dtype),
        "pools": pools,
        "limit_ok": bool(limit_ok),
        "input_unchanged": float(image.sum(dtype=np.float64)) == checksum,
    }
    Path(args.out).write_text(json.dumps(record))


# --------------------------------------------------------------------------------------
# Probe: versions and capabilities of the environment, found out in a child process so
# that the parent never imports the codes under test
# --------------------------------------------------------------------------------------
def probe_main(args) -> None:
    from importlib import metadata

    names = (
        "dfcosmic",
        "torch",
        "numpy",
        "scipy",
        "astropy",
        "astroscrappy",
        "lacosmic",
        "threadpoolctl",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    info = {"versions": versions, "available": {}, "gpu": None}
    try:
        import torch
        from dfcosmic.utils import cpp_median_available

        versions["torch"] = torch.__version__
        info["available"]["dfcosmic_torch"] = True
        info["available"]["dfcosmic_cpp"] = bool(cpp_median_available())
        info["available"]["dfcosmic_gpu"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": round(properties.total_memory / 2**30, 1),
                "cuda": torch.version.cuda,
            }
    except ImportError:
        for config in ("dfcosmic_torch", "dfcosmic_cpp", "dfcosmic_gpu"):
            info["available"][config] = False
    try:
        from astroscrappy import detect_cosmics  # noqa: F401

        astroscrappy_ok = True
    except ImportError:
        astroscrappy_ok = False
    info["available"]["astroscrappy_sepmed_false"] = astroscrappy_ok
    info["available"]["astroscrappy_sepmed_true"] = astroscrappy_ok
    try:
        from lacosmic import remove_cosmics  # noqa: F401

        info["available"]["lacosmic"] = True
    except ImportError:
        info["available"]["lacosmic"] = False
    Path(args.out).write_text(json.dumps(info))


# --------------------------------------------------------------------------------------
# System information
# --------------------------------------------------------------------------------------
def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _command(*cmd: str) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _cpu_info() -> dict:
    cpuinfo = _read("/proc/cpuinfo")
    if cpuinfo is None:  # not Linux
        return {
            "model": _command("sysctl", "-n", "machdep.cpu.brand_string")
            or platform.processor(),
            "physical_cores": None,
        }
    model, cores = None, set()
    physical_id = None
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "model name" and model is None:
            model = value
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            cores.add((physical_id, value))
    return {"model": model, "physical_cores": len(cores) or None}


def _ram_gb() -> float | None:
    meminfo = _read("/proc/meminfo")
    if meminfo is not None:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 1)
    memsize = _command("sysctl", "-n", "hw.memsize")
    return round(int(memsize) / 2**30, 1) if memsize else None


def _os_name() -> str:
    # Inside a Flatpak sandbox, /run/host describes the real operating system
    for path in ("/run/host/os-release", "/etc/os-release"):
        text = _read(path)
        if text:
            for line in text.splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip('"')
    return platform.system()


def source_checksum() -> int | None:
    """CRC of the dfcosmic source files, to tell which code a result belongs to."""
    root = HERE.parent
    files = sorted((root / "src" / "dfcosmic").glob("*.py"))
    files += sorted((root / "csrc").glob("*.cpp"))
    if not files:
        return None
    checksum = 0
    for path in files:
        checksum = zlib.crc32(path.read_bytes(), checksum)
    return checksum


def system_info() -> dict:
    cpu = _cpu_info()
    nvidia = _read("/proc/driver/nvidia/version")
    commit = _command("git", "-C", str(HERE), "rev-parse", "--short=7", "HEAD")
    dirty = _command("git", "-C", str(HERE), "status", "--porcelain", "--", "../src")
    return {
        "cpu": cpu["model"],
        "physical_cores": cpu["physical_cores"],
        "logical_cpus": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_boost": _read("/sys/devices/system/cpu/cpufreq/boost"),
        "os": _os_name(),
        "kernel": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "nvidia_driver": nvidia.splitlines()[0] if nvidia else None,
        "dfcosmic_commit": commit,
        "dfcosmic_src_modified": bool(dirty) if dirty is not None else None,
        "dfcosmic_source_checksum": source_checksum(),
    }


def probe() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "probe.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "_probe", "--out", out]
        subprocess.run(cmd, check=True, env=child_env(1))
        return json.loads(out.read_text())


# --------------------------------------------------------------------------------------
# Running the jobs
# --------------------------------------------------------------------------------------
def child_env(threads: int) -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_SCRUB_ENV_PREFIXES)
    }
    for name in _THREAD_ENV:
        env[name] = str(threads)
    return env


def _busy_cpu_seconds() -> float | None:
    """CPU time used by everything on the machine so far (Linux only)."""
    stat = _read("/proc/stat")
    if stat is None:
        return None
    fields = [int(value) for value in stat.splitlines()[0].split()[1:]]
    user, nice, system, _idle, _iowait, irq, softirq, steal = fields[:8]
    return (user + nice + system + irq + softirq + steal) / os.sysconf("SC_CLK_TCK")


def _children_cpu_seconds() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def wait_for_quiet(max_background: float, patience: float = 300.0) -> None:
    """Wait, for a limited time, until other processes leave the machine alone."""
    deadline = time.perf_counter() + patience
    while True:
        busy, start = _busy_cpu_seconds(), time.perf_counter()
        if busy is None:
            return
        time.sleep(2)
        background = (_busy_cpu_seconds() - busy) / (time.perf_counter() - start)
        if background <= max_background or time.perf_counter() > deadline:
            return
        time.sleep(10)


def run_job(config, threads, niter, frame, repeats, timeout, image="synthetic") -> dict:
    """Time one configuration at one thread count in a fresh process."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "record.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "_worker"]
        cmd += ["--config", config, "--threads", str(threads), "--niter", str(niter)]
        cmd += ["--frame", str(frame), "--repeats", str(repeats), "--out", str(out)]
        cmd += ["--image", image]

        busy, children = _busy_cpu_seconds(), _children_cpu_seconds()
        start = time.perf_counter()
        try:
            done = subprocess.run(
                cmd,
                env=child_env(threads),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            error = done.stderr[-2000:] if done.returncode != 0 else None
        except subprocess.TimeoutExpired:
            error = f"timed out after {timeout} s"
        wall = time.perf_counter() - start

        if error is None and not out.exists():
            error = "the worker did not write a record"
        if error is not None:
            return {"config": config, "threads": threads, "error": error}
        record = json.loads(out.read_text())

    # CPU time used by everything else on the machine while this job ran
    background = None
    if busy is not None:
        others = (_busy_cpu_seconds() - busy) - (_children_cpu_seconds() - children)
        background = max(0.0, others) / wall
    record["background_busy_cpus"] = background
    return record


def _save(results: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=1) + "\n")
    os.replace(tmp, path)


def threads_for(config: str, threads) -> list[int]:
    threads = sorted(threads)
    if CONFIGS[config]["threads"] == "ends":
        return sorted({threads[0], threads[-1]})
    return threads


def run_main(args) -> None:
    quick = args.quick
    shape = tuple(args.shape or ((1001, 1001) if quick else FRAME_SHAPE))
    threads = args.threads or ([1, 2] if quick else list(THREADS))
    rounds = args.rounds or (1 if quick else 5)
    repeats = args.repeats or (1 if quick else 3)
    frame_dir = Path(
        args.frame_dir or Path(tempfile.gettempdir()) / "dfcosmic-benchmark"
    )
    output = Path(
        args.output or (frame_dir / "quick_results.json" if quick else DEFAULT_RESULTS)
    )
    image = args.image
    key = f"niter{args.niter}" if image == "synthetic" else f"{image}_niter{args.niter}"

    environment = probe()
    configs = args.configs or list(CONFIGS)
    for config in configs:
        if not environment["available"].get(config):
            print(f"Skipping {config}: not available in this environment")
    configs = [config for config in configs if environment["available"].get(config)]

    settings = {
        "image": image,
        "shape": list(shape),
        "dtype": "float32",
        "seed": SEED,
        "params": IMAGES[image]["params"],
        "niter": args.niter,
        "threads": sorted(threads),
        "rounds": rounds,
        "repeats": repeats,
        "warmup_calls": 1,
        "shuffle_seed": args.shuffle_seed,
        "max_background_cpus": args.max_background,
    }
    results = json.loads(output.read_text()) if output.exists() else {"schema": 1}
    results.setdefault("experiments", {})
    experiment = results["experiments"].get(key)
    fresh = args.fresh or quick
    if experiment is not None and (fresh or experiment["settings"] != settings):
        if not fresh:
            sys.exit(
                f"{output} already holds a '{key}' run with other settings. "
                "Use --fresh to replace it, or --output to write elsewhere."
            )
        experiment = None
    if experiment is None:
        experiment = {
            "settings": settings,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "system": system_info(),
            "gpu": environment["gpu"],
            "versions": environment["versions"],
            "configs": {name: CONFIGS[name] for name in configs},
            "runs": [],
        }
        results["experiments"][key] = experiment
    experiment["configs"].update({name: CONFIGS[name] for name in configs})

    frame = frame_file(shape, frame_dir, image)
    done = {
        (run["config"], run["threads"], run["round"])
        for run in experiment["runs"]
        if "error" not in run
    }
    experiment["runs"] = [run for run in experiment["runs"] if "error" not in run]
    jobs = [(c, t) for c in configs for t in threads_for(c, threads)]

    for round_index in range(rounds):
        order = list(jobs)
        random.Random(args.shuffle_seed + round_index).shuffle(order)
        for config, n in order:
            if (config, n, round_index) in done:
                continue
            for attempt in range(args.attempts):
                wait_for_quiet(args.max_background)
                record = run_job(
                    config, n, args.niter, frame, repeats, args.timeout, image
                )
                background = record.get("background_busy_cpus")
                if background is None or background <= args.max_background:
                    break
                print(
                    f"{config}, {n} thread(s): {background:.1f} CPUs were busy with "
                    "other work"
                    + (", repeating" if attempt + 1 < args.attempts else ""),
                    flush=True,
                )
                time.sleep(args.pause)
            record["round"] = round_index
            experiment["runs"].append(record)
            experiment["finished"] = datetime.datetime.now().isoformat(
                timespec="seconds"
            )
            _save(results, output)

            prefix = f"[round {round_index + 1}/{rounds}] {config}, {n} thread(s):"
            if "error" in record:
                print(prefix, "FAILED\n", record["error"], flush=True)
                continue
            cores = max(c / t for c, t in zip(record["cpu_s"], record["times_s"]))
            background = record["background_busy_cpus"]
            print(
                prefix,
                f"median {statistics.median(record['times_s']):.3f} s,",
                f"first call {record['first_call_s']:.3f} s,",
                f"cores busy {cores:.2f},",
                "background "
                + ("n/a" if background is None else f"{background:.2f} CPUs"),
                "" if record["limit_ok"] else " !! THREAD LIMIT NOT APPLIED",
                flush=True,
            )
    print(f"Results written to {output}")


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def load_results(path=DEFAULT_RESULTS) -> dict:
    return json.loads(Path(path).read_text())


def summarise(experiment: dict) -> dict:
    """Statistics per (configuration, threads) over all timed calls of all rounds."""
    grouped: dict = {}
    for run in experiment["runs"]:
        if "error" in run:
            continue
        grouped.setdefault((run["config"], run["threads"]), []).append(run)

    summary = {}
    for (config, threads), runs in grouped.items():
        times = [t for run in runs for t in run["times_s"]]
        cpu = [c for run in runs for c in run["cpu_s"]]
        background = [run["background_busy_cpus"] for run in runs]
        median = statistics.median(times)
        summary[(config, threads)] = {
            "median": median,
            "min": min(times),
            "max": max(times),
            "n": len(times),
            "spread": (max(times) - min(times)) / median,
            "first_call": statistics.median(run["first_call_s"] for run in runs),
            "cores_busy": max(c / t for c, t in zip(cpu, times)),
            "limit_ok": all(run["limit_ok"] for run in runs),
            "input_unchanged": all(run["input_unchanged"] for run in runs),
            "background": None if None in background else max(background),
            "passes": sorted({run["passes"] for run in runs}),
            "n_flagged": sorted({run["n_flagged"] for run in runs}),
            "n_injected": runs[0]["n_injected"],
            "n_injected_flagged": sorted({run["n_injected_flagged"] for run in runs}),
        }
    return summary


def _seconds(value: float) -> str:
    """Three significant figures."""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _unique(values: list) -> str:
    return "/".join(str(value) for value in values)


def _measured(experiment: dict) -> tuple[list, list]:
    summary = summarise(experiment)
    configs = [c for c in CONFIGS if any(key[0] == c for key in summary)]
    threads = sorted({key[1] for key in summary})
    return configs, threads


def timing_table(experiment: dict) -> str:
    """Markdown table of the median runtime, with the min-max range in brackets."""
    summary = summarise(experiment)
    configs, threads = _measured(experiment)
    header = ["Configuration"] + [f"{n} thread{'s' if n > 1 else ''}" for n in threads]
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * len(threads)]
    for config in configs:
        cells = []
        for n in threads:
            stats = summary.get((config, n))
            if stats is None:
                cells.append("–")
            else:
                cells.append(
                    f"{_seconds(stats['median'])} "
                    f"({_seconds(stats['min'])}–{_seconds(stats['max'])})"
                )
        lines.append(f"| {CONFIGS[config]['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ratio_table(experiment: dict, reference="astroscrappy_sepmed_false") -> str:
    """Markdown table of runtime relative to a reference configuration."""
    summary = summarise(experiment)
    configs, threads = _measured(experiment)
    header = ["Configuration"] + [f"{n} thread{'s' if n > 1 else ''}" for n in threads]
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * len(threads)]
    for config in configs:
        cells = []
        for n in threads:
            stats, ref = summary.get((config, n)), summary.get((reference, n))
            if stats is None or ref is None:
                cells.append("–")
            else:
                cells.append(f"{stats['median'] / ref['median']:.2f}")
        lines.append(f"| {CONFIGS[config]['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def speedup_table(before: dict, after: dict, configs=None) -> str:
    """Markdown table comparing two runs of the same configurations."""
    old, new = summarise(before), summarise(after)
    keys = [
        key
        for key in sorted(new, key=lambda key: (list(CONFIGS).index(key[0]), key[1]))
        if key in old and (configs is None or key[0] in configs)
    ]
    lines = [
        "| Configuration | Threads | Before (s) | After (s) | Times faster |",
        "|---|---:|---:|---:|---:|",
    ]
    for config, threads in keys:
        was, now = old[(config, threads)]["median"], new[(config, threads)]["median"]
        lines.append(
            f"| {CONFIGS[config]['label']} | {threads} | {_seconds(was)} "
            f"| {_seconds(now)} | {was / now:.2f} |"
        )
    return "\n".join(lines)


_FUNCTIONS = {
    "dfcosmic": "dfcosmic.lacosmic",
    "astroscrappy": "astroscrappy.detect_cosmics",
    "lacosmic": "lacosmic.remove_cosmics",
}


def calls_table(experiment: dict) -> str:
    """Markdown table of the exact call made for every configuration."""
    settings = experiment["settings"]
    configs, _ = _measured(experiment)
    lines = ["| Configuration | Call |", "|---|---|"]
    for config in configs:
        kwargs = call_kwargs(config, settings["params"], settings["niter"], "n")
        arguments = ", ".join(
            f"{key}={value!r}".replace("'n'", "n") for key, value in kwargs.items()
        )
        function = _FUNCTIONS[CONFIGS[config]["package"]]
        lines.append(
            f"| {CONFIGS[config]['label']} | `{function}(image, {arguments})` |"
        )
    return "\n".join(lines)


def _injected(stats: dict) -> str:
    total = next(iter(stats.values()))["n_injected"]
    if total == 0:  # an image without injected cosmic rays
        return "–"
    found = sorted({n for s in stats.values() for n in s["n_injected_flagged"]})
    return f"{_unique(found)} of {total}"


def checks_table(experiment: dict) -> str:
    """Markdown table of the self-checks recorded with the timings."""
    summary = summarise(experiment)
    configs, _ = _measured(experiment)
    lines = [
        "| Configuration | Passes run | Pixels flagged | Injected cosmic rays flagged "
        "| Thread limit applied | Most cores busy / threads | Largest spread "
        "| First call, 1 thread (s) |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for config in configs:
        stats = {n: s for (c, n), s in summary.items() if c == config}
        first = stats[min(stats)]
        lines.append(
            f"| {CONFIGS[config]['label']} "
            f"| {_unique(sorted({p for s in stats.values() for p in s['passes']}))} "
            f"| {_unique(sorted({p for s in stats.values() for p in s['n_flagged']}))} "
            f"| {_injected(stats)} "
            f"| {'yes' if all(s['limit_ok'] for s in stats.values()) else 'NO'} "
            f"| {max(s['cores_busy'] / n for n, s in stats.items()):.2f} "
            f"| {100 * max(s['spread'] for s in stats.values()):.1f}% "
            f"| {_seconds(first['first_call'])} |"
        )
    return "\n".join(lines)


def _cpu_name(system: dict) -> str:
    # "AMD Ryzen 9 9950X 16-Core Processor" -> "AMD Ryzen 9 9950X"
    return re.sub(r"\s+\d+-Core Processor$", "", system["cpu"] or "unknown CPU")


def provenance(experiment: dict) -> str:
    """One paragraph stating what was measured, how, on what, and with which versions."""
    settings, system = experiment["settings"], experiment["system"]
    versions, gpu = experiment["versions"], experiment["gpu"]
    params = ", ".join(f"{key}={value:g}" for key, value in settings["params"].items())
    calls = settings["rounds"] * settings["repeats"]
    hardware = _cpu_name(system)
    if system.get("physical_cores"):
        hardware += (
            f" ({system['physical_cores']} cores, {system['logical_cpus']} threads)"
        )
    if gpu:
        hardware += f", {gpu['name']} ({gpu['memory_gb']:.0f} GB)"
    commit = system.get("dfcosmic_commit")
    if commit and system.get("dfcosmic_src_modified"):
        commit += " with uncommitted changes"
    software = ", ".join(
        f"{name} {versions[name]}"
        + (f" (commit {commit})" if name == "dfcosmic" and commit else "")
        for name in ("dfcosmic", "torch", "astroscrappy", "lacosmic", "numpy", "scipy")
        if versions.get(name)
    )
    description = IMAGES[settings.get("image", "synthetic")]["description"]
    return (
        f"Image: {description}; {settings['shape'][0]} × {settings['shape'][1]} "
        f"pixels, {settings['dtype']}. Parameters: niter={settings['niter']}, {params}. "
        f"Statistic: median of {calls} timed calls ({settings['rounds']} processes × "
        f"{settings['repeats']} calls, each process after one warm-up call); "
        "ranges are minimum to maximum. "
        f"Hardware: {hardware}. "
        f"Software: {system['os']} ({system['kernel']}), Python {system['python']}, "
        f"{software}."
    )


def find_problems(experiment: dict) -> list[str]:
    """Problems that make a measured point unreliable."""
    problems = [
        f"{run['config']} at {run['threads']} thread(s) failed: {run['error'][-300:]}"
        for run in experiment["runs"]
        if "error" in run
    ]
    expected = experiment["settings"]["rounds"] * experiment["settings"]["repeats"]
    for (config, threads), stats in sorted(summarise(experiment).items()):
        where = f"{config} at {threads} thread(s)"
        if not stats["limit_ok"]:
            problems.append(f"{where}: a thread pool was not limited to {threads}")
        if stats["cores_busy"] > 1.1 * threads + 0.1 and config != "dfcosmic_gpu":
            problems.append(f"{where}: kept {stats['cores_busy']:.2f} cores busy")
        if stats["n"] != expected:
            problems.append(f"{where}: {stats['n']} timed calls instead of {expected}")
        if not stats["input_unchanged"]:
            problems.append(f"{where}: the input image was modified")
        limit = experiment["settings"].get("max_background_cpus", 3.0)
        if stats["background"] is not None and stats["background"] > limit:
            problems.append(
                f"{where}: other processes kept {stats['background']:.1f} CPUs busy"
            )
    return problems


# Colours from the validated default palette of the dataviz guidelines: one hue per
# package, marker shape and line style for the variant. The separable median is the
# only dotted line, because it is the only different algorithm.
_THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "hue": {
            "dfcosmic": "#2a78d6",
            "astroscrappy": "#eb6834",
            "lacosmic": "#1baf7a",
        },
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "hue": {
            "dfcosmic": "#3987e5",
            "astroscrappy": "#d95926",
            "lacosmic": "#199e70",
        },
    },
}
_MARKER_SIZE = 9
_MARKS = {
    "dfcosmic_torch": ("o", "-"),
    "dfcosmic_cpp": ("s", "-"),
    "dfcosmic_gpu": ("^", "-"),
    "astroscrappy_sepmed_false": ("D", "-"),
    "astroscrappy_sepmed_true": ("v", ":"),
    "lacosmic": ("h", "-"),
}


def plot(experiment: dict, dark: bool = False):
    """Runtime against the number of CPU threads, for every configuration."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    theme = _THEMES["dark" if dark else "light"]
    summary = summarise(experiment)
    configs, threads = _measured(experiment)
    gpu = experiment["gpu"]

    fig = Figure(figsize=(8.6, 5.8), facecolor=theme["surface"])
    ax = fig.subplots()
    ax.set_facecolor(theme["surface"])

    for config in configs:
        measured = [n for n in threads if (config, n) in summary]
        median = np.array([summary[(config, n)]["median"] for n in measured])
        low = np.array([summary[(config, n)]["min"] for n in measured])
        high = np.array([summary[(config, n)]["max"] for n in measured])
        color = theme["hue"][CONFIGS[config]["package"]]
        marker, linestyle = _MARKS[config]
        label = CONFIGS[config]["label"]
        if config == "dfcosmic_gpu" and gpu:
            label += f" ({gpu['name'].replace('NVIDIA GeForce ', '')})"
        ax.errorbar(
            measured,
            median,
            yerr=[median - low, high - median],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            zorder=2,
        )
        ax.plot(
            measured,
            median,
            color=color,
            linewidth=2,
            linestyle=linestyle,
            marker=marker,
            markersize=_MARKER_SIZE,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
            solid_capstyle="round",
            label=label,
            zorder=3,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(threads))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    values = [stats[key] for stats in summary.values() for key in ("min", "max")]
    low, high = min(values) / 1.35, max(values) * 1.35
    ax.set_ylim(low, high)
    ticks = [
        t
        for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200)
        if low <= t <= high
    ]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    ax.grid(True, which="major", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["ink_secondary"], labelsize=11, length=0)
    ax.set_xlabel("CPU threads", color=theme["ink_secondary"], fontsize=12)
    ax.set_ylabel("Runtime per image (s)", color=theme["ink_secondary"], fontsize=12)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        labelcolor=theme["ink"],
        handlelength=2.6,
        columnspacing=1.6,
    )
    legend.set_zorder(4)

    fig.subplots_adjust(left=0.09, right=0.975, top=0.975, bottom=0.27)
    return fig


def figure_caption(experiment: dict) -> str:
    """Caption for the figure drawn by ``plot``: what was measured, and how."""
    settings = experiment["settings"]
    summary = summarise(experiment)
    calls = settings["rounds"] * settings["repeats"]
    spread = 100 * max(stats["spread"] for stats in summary.values())

    # The range of the timed calls is drawn as a bar. Check whether any bar is long
    # enough to be seen behind its marker, so that the caption does not promise bars
    # that cannot be seen.
    fig = plot(experiment)
    ax = fig.axes[0]
    marker_radius = _MARKER_SIZE / 72 * fig.dpi / 2
    longest_bar = 0.0
    for (_, n), stats in summary.items():
        pixels = ax.transData.transform(
            [(n, stats["min"]), (n, stats["median"]), (n, stats["max"])]
        )[:, 1]
        longest_bar = max(longest_bar, pixels[1] - pixels[0], pixels[2] - pixels[1])
    if longest_bar > 1.25 * marker_radius:
        ranges = "the bars show the minimum and the maximum"
    else:
        ranges = (
            f"their range, at most {spread:.0f}% of the median, is smaller than the "
            "markers"
        )
    return (
        f"Runtime per {settings['shape'][0]} × {settings['shape'][1]} image against "
        f"the number of CPU threads, with niter={settings['niter']} and the same "
        f"parameters in every code. Each point is the median of {calls} timed calls; "
        f"{ranges}."
    )


def report_main(args) -> None:
    results = load_results(args.input)
    for key, experiment in sorted(results["experiments"].items()):
        print(f"\n## {key}\n")
        print(provenance(experiment), "\n")
        print("Median runtime in seconds (minimum–maximum):\n")
        print(timing_table(experiment), "\n")
        print(
            "Runtime relative to astroscrappy with a true median (above 1 is slower):\n"
        )
        print(ratio_table(experiment), "\n")
        print("Checks:\n")
        print(checks_table(experiment), "\n")
        problems = find_problems(experiment)
        print("Warnings:" if problems else "No warnings.")
        for problem in problems:
            print(f"- {problem}")

    if args.no_figures:
        return
    experiment = results["experiments"].get(args.figure)
    if experiment is None:
        sys.exit(f"no '{args.figure}' run in {args.input}; cannot draw the figure")
    outdir = Path(args.outdir)
    for dark, name in ((False, "comparison.png"), (True, "comparison_dark.png")):
        fig = plot(experiment, dark=dark)
        fig.savefig(outdir / name, dpi=200, facecolor=fig.get_facecolor())
        print(f"Wrote {outdir / name}")
    print("Caption:", figure_caption(experiment))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="time every configuration")
    run.add_argument("--niter", type=int, default=1)
    run.add_argument(
        "--image",
        choices=list(IMAGES),
        default="synthetic",
        help="test image: few cosmic rays (synthetic) or crowded (hst)",
    )
    run.add_argument("--threads", type=int, nargs="+")
    run.add_argument("--shape", type=int, nargs=2, metavar=("ROWS", "COLUMNS"))
    run.add_argument("--rounds", type=int, help="fresh processes per point (default 5)")
    run.add_argument("--repeats", type=int, help="timed calls per process (default 3)")
    run.add_argument("--configs", nargs="+", choices=list(CONFIGS))
    run.add_argument("--output", help=f"results file (default {DEFAULT_RESULTS})")
    run.add_argument("--frame-dir", help="where the test image is kept")
    run.add_argument("--shuffle-seed", type=int, default=0)
    run.add_argument("--timeout", type=float, default=3600, help="seconds per process")
    run.add_argument(
        "--max-background",
        type=float,
        default=MAX_BACKGROUND_CPUS,
        help="repeat a job if other processes kept more CPUs than this busy",
    )
    run.add_argument("--attempts", type=int, default=3, help="tries per job")
    run.add_argument("--pause", type=float, default=20, help="seconds before a retry")
    run.add_argument("--fresh", action="store_true", help="discard an existing run")
    run.add_argument(
        "--quick",
        action="store_true",
        help="1001 x 1001 image, 1 and 2 threads, written to a temporary file",
    )
    run.set_defaults(function=run_main)

    report = commands.add_parser("report", help="tables and figures from the results")
    report.add_argument("--input", default=str(DEFAULT_RESULTS))
    report.add_argument("--outdir", default=str(HERE))
    report.add_argument("--figure", default="niter1", help="run shown in the figure")
    report.add_argument("--no-figures", action="store_true")
    report.set_defaults(function=report_main)

    worker = commands.add_parser("_worker")
    worker.add_argument("--config", required=True, choices=list(CONFIGS))
    worker.add_argument("--threads", type=int, required=True)
    worker.add_argument("--niter", type=int, required=True)
    worker.add_argument("--image", choices=list(IMAGES), default="synthetic")
    worker.add_argument("--frame", required=True)
    worker.add_argument("--repeats", type=int, required=True)
    worker.add_argument("--out", required=True)
    worker.set_defaults(function=worker_main)

    probe_parser = commands.add_parser("_probe")
    probe_parser.add_argument("--out", required=True)
    probe_parser.set_defaults(function=probe_main)

    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
