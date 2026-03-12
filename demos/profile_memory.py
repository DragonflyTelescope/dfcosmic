import argparse
import time

import numpy as np

from dfcosmic import lacosmic


def make_image(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.normal(100.0, 5.0, size=shape).astype(np.float32)

    # Add a few compact sources.
    y, x = np.ogrid[: shape[0], : shape[1]]
    for x0, y0, amp, sigma in [
        (shape[1] * 0.2, shape[0] * 0.3, 300.0, 2.0),
        (shape[1] * 0.6, shape[0] * 0.7, 500.0, 3.0),
        (shape[1] * 0.8, shape[0] * 0.2, 250.0, 1.5),
    ]:
        rsq = (x - x0) ** 2 + (y - y0) ** 2
        image += amp * np.exp(-rsq / (2.0 * sigma**2)).astype(np.float32)

    # Add synthetic cosmic rays.
    n_cr = max(10, (shape[0] * shape[1]) // 500_000)
    cr_y = rng.integers(0, shape[0], size=n_cr)
    cr_x = rng.integers(0, shape[1], size=n_cr)
    image[cr_y, cr_x] += rng.uniform(1_000.0, 10_000.0, size=n_cr).astype(np.float32)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory profiling harness for dfcosmic."
    )
    parser.add_argument("--height", type=int, default=4176)
    parser.add_argument("--width", type=int, default=6248)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--niter", type=int, default=1)
    parser.add_argument("--sigclip", type=float, default=4.5)
    parser.add_argument("--sigfrac", type=float, default=0.5)
    parser.add_argument("--objlim", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--readnoise", type=float, default=5.0)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--no-cpp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image = make_image((args.height, args.width), seed=args.seed)
    print(
        f"Profiling lacosmic on shape={image.shape}, dtype={image.dtype}, "
        f"device={args.device}, niter={args.niter}"
    )

    start = time.perf_counter()
    clean, mask = lacosmic(
        image=image,
        sigclip=args.sigclip,
        sigfrac=args.sigfrac,
        objlim=args.objlim,
        niter=args.niter,
        gain=args.gain,
        readnoise=args.readnoise,
        device=args.device,
        cpu_threads=args.cpu_threads,
        use_cpp=not args.no_cpp,
        rss_debug=True,
    )
    elapsed = time.perf_counter() - start
    print(
        f"Done in {elapsed:.2f}s. clean_shape={clean.shape}, "
        f"mask_pixels={int(mask.sum())}"
    )


if __name__ == "__main__":
    main()
