import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from astroscrappy import detect_cosmics
from lacosmic.core import lacosmic
from threadpoolctl import threadpool_limits

from dfcosmic import lacosmic as df_lacosmic

num_threads = [1, 4]


# Make a simple Gaussian function for testing purposes
def gaussian(image_shape, x0, y0, brightness, fwhm):
    x = np.arange(image_shape[1])
    y = np.arange(image_shape[0])
    x2d, y2d = np.meshgrid(x, y)

    sig = fwhm / 2.35482

    normfactor = brightness / 2.0 / np.pi * sig**-2.0
    exponent = -0.5 * sig**-2.0
    exponent *= (x2d - x0) ** 2.0 + (y2d - y0) ** 2.0

    return normfactor * np.exp(exponent)


def make_fake_data(size=(6000, 4000)):
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
    np.random.seed(200)

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


imdata, expected_crmask = make_fake_data()

dfcosmic_cpu = []
for n_threads in num_threads:
    torch.set_num_threads(n_threads)
    start_dfcosmic_cpu = time.perf_counter()
    clean, crmask = df_lacosmic(
        image=imdata,
        objlim=2,
        sigfrac=1,
        sigclip=6,
        gain=1,
        readnoise=10,
        niter=1,
        device="cpu",
        debug_backend=True,
        cpu_threads=n_threads,
    )

    elapsed_dfcosmic_cpu = time.perf_counter() - start_dfcosmic_cpu
    print(elapsed_dfcosmic_cpu)
    dfcosmic_cpu.append(elapsed_dfcosmic_cpu)

astroscrappy_sepmed_false = []
for n_threads in num_threads:
    with threadpool_limits(limits=n_threads):
        start_astroscrappy = time.perf_counter()
        _, _ = detect_cosmics(
            imdata,
            readnoise=10.0,
            gain=1.0,
            sigclip=6,
            sigfrac=1.0,
            objlim=2,
            sepmed=False,
        )
        elapsed_astroscrappy = time.perf_counter() - start_astroscrappy
        print(elapsed_astroscrappy)
        astroscrappy_sepmed_false.append(elapsed_astroscrappy)

colors = [
    "#648FFF",  # Blue
    "#785EF0",  # Purple
    "#00B4D8",  # Cyan
    "#DC267F",  # Magenta
    "#FE6100",  # Orange
    "#FFB000",  # Yellow
]

fig = plt.figure(figsize=(16, 16))
plt.plot(num_threads, dfcosmic_cpu, label="dfcosmic (CPU)", c=colors[0])


plt.plot(
    num_threads,
    astroscrappy_sepmed_false,
    label="astroscrappy (sepmed=False)",
    c=colors[3],
)


plt.legend()
plt.show()
