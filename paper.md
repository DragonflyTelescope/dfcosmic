---
title: 'dfcosmic: A Python package for cosmic ray removal'
tags:
  - Python
  - astronomy
  - lacosmic
  - PyTorch
  - Dragonfly Telephoto Array
authors:
  - name: Carter Lee Rhea
    orcid: 0000-0003-2001-1076
    affiliation: "1, 2" 
  - name: Pieter van Dokkum
    affiliation: "1, 3"
  - name: Steven R. Janssens
    orcid: 0000-0003-0327-3322
    affiliation: 1
  - name: Imad Pasha
    affiliation: "1, 3"
  - name: Roberto Abraham
    affiliation: "1, 4, 5"
  - name: William P. Bowman
    orcid: 0000-0003-4381-5245
    affiliation: "1, 3"
  - name: Deborah Lokhorst
    affiliation: "1, 6"
  - name: Seery Chen
    affiliation: "1, 4, 5"

affiliations:
 - name: Dragonfly Focused Research Organization, 150 Washington Avenue, Santa Fe, 87501, NM, USA
   index: 1
 - name: Centre de Recherche en Astrophysique du Québec (CRAQ), Québec, QC G1V 0A6, Canada
   index: 2
 - name: Astronomy Department, Yale University, 219 Prospect St, New Haven, CT 06511, USA
   index: 3
 - name: David A. Dunlap Department of Astronomy & Astrophysics, University of Toronto, 50 St. George Street, Toronto, ON M5S 3H4, Canada
   index: 4
 - name: Dunlap Institute for Astronomy & Astrophysics, University of Toronto, 50 St. George Street, Toronto, ON M5S 3H4, Canada
   index: 5
 - name: NRC Herzberg Astronomy & Astrophysics Research Centre, 5071 West Saanich Road, Victoria, BC V9E 2E7, Canada
   index: 6
date: 01 February 2026
bibliography: dfcosmic.bib

---

# Summary

Astronomical images often show sharp features that are caused by cosmic ray (CR) hits, hot pixels, or non-Gaussian noise. L.A.Cosmic [@van_dokkum_cosmic-ray_2001] is a widely used edge detection algorithm that identifies and replaces such features. Here we describe `dfcosmic`, a direct python port of L.A.Cosmic utilizing PyTorch, with an optional C++ median filter, to enable efficient performance on both CPUs and GPUs. The code was developed for the MOTHRA array, which is projected to produce more than 1000 large format CMOS images every 15 minutes. With matched settings and two threads per image, `dfcosmic` with its C++ median filter takes 30 to 40% less time than the fastest existing Python implementation that uses a true median filter, and on a GPU it takes less than half a second per image.

# Statement of need
The Modular Optical Telephoto Hyperspectral Robotic Array (MOTHRA) uses CMOS sensors rather than traditional CCDs. Modern CMOS detectors have extraordinarily low noise but suffer from a relatively high number of hot pixels and non-Gaussian noise ("salt-and-pepper"; [@alarcon_scientific_2023]).
Therefore, the data reduction pipeline for MOTHRA requires rapid bright pixel identification and removal for tens of thousands of images every night using only a single core (2 threads) per frame.
Although several implementations of L.A.Cosmic [@van_dokkum_cosmic-ray_2001] exist such as `lacosmic` [@bradley_larrybradleylacosmic_2025] and `astroscrappy` [@robitaille_astropyastroscrappy_2025], these implementations either deviate from the original algorithm in order to achieve computational gains or do not run fast enough for our usage. Importantly, experiments on the preliminary data taken by MOTHRA have demonstrated that it is crucial to use the original implementation (notably a true median filter rather than a separable median filter) in order to capture all the CRs (or hot pixels or salt-and-pepper non-Gaussian noise) without accidentally removing bright stars. `dfcosmic` has already been adopted in the nightly reduction pipeline for MOTHRA.


# State of the field
More broadly modern high frequency observatories are taking thousands of images each night; Therefore, it is necessary to have a fast, and reliable, implementation of the algorithm to reduce all the data in a reasonable amount of time.
Although several methods for detecting CRs in astronomical images have been proposed (i.e. @zhang_deepcr_2020, @pych_fast_2003, @xu_cosmic-conn_2023), the most widely use algorithm is the L.A.Cosmic algorithm [@van_dokkum_cosmic-ray_2001].
There currently exist other cosmic ray removal codes in Python based on the L.A.Cosmic algorithm; notably  `lacosmic` [@bradley_larrybradleylacosmic_2025] and `astroscrappy` [@robitaille_astropyastroscrappy_2025].  Moreover, although the current data reduction infrastructure only supports CPU computing, we wish to have a package that will eventually be able to run rapidly on a GPU. 

In light of these considerations, we have developed `dfcosmic`. We benchmark it against the `astroscrappy` and `lacosmic` implementations. Notably, `astroscrappy`'s default run configuration uses a separable median filter which is a non-negligible departure from the original algorithm in order to speed up the algorithm. We run `astroscrappy` in our benchmarking with and without the separable median filter active. We stress that the use of a separable median filter can result in the incorrect removal of cosmic rays by incorrectly classifying the center of near-saturated or saturated stars as cosmic rays. We show the results of the three different versions of `dfcosmic`: CPU with torch only, CPU with C++ optimization, and GPU. We discuss these three different versions in the software design section of this paper.

We run the codes on the mock data used for testing by [`astroscrappy`](https://github.com/astropy/astroscrappy/blob/main/astroscrappy/tests/fake_data.py) with a typically sized frame for MOTHRA (4000x6500). Each option was run employing 1, 2, 4, 8, and 16 threads.
The GPU used in this test was an NVIDIA GeForce RTX 5060 Ti 16GB while the CPU was an AMD Ryzen 9 9950X 16-Core Processor.
All codes were given the same image and the same parameters, and ran a single iteration. Each configuration was timed in its own process, with one warm-up call followed by three timed calls, and this was repeated in five processes; we report the median of the fifteen timed calls. We used `dfcosmic` 0.2.0, `astroscrappy` 1.3.0, `lacosmic` 1.4.0 and PyTorch 2.14.1. The benchmark script, its results and a notebook that displays them are in the `demos` directory of the repository.

![\label{fig:comparison} Runtime per 4000x6500 image as a function of the number of CPU threads, for a single iteration with the same parameters in every code. Each point is the median of fifteen timed calls; their range, at most 12% of the median, is smaller than the markers. `astroscrappy` with `sepmed=True` uses a separable median filter, which is a different algorithm. Measured on an AMD Ryzen 9 9950X CPU and an NVIDIA GeForce RTX 5060 Ti GPU.](demos/comparison.png)

Among the implementations with a true median filter, `dfcosmic` with its C++ median filter is the fastest at every number of threads (\autoref{fig:comparison}): it takes 44% less time than `astroscrappy` on one thread (15.3 s against 27.1 s), 40% less on two (8.5 s against 14.2 s) and 23% less on 16 (2.2 s against 2.8 s). Without the C++ median filter, `dfcosmic` is faster than `astroscrappy` on one to four threads, about level on eight, and 16% slower on 16. On the GPU an image takes 0.33 s. `astroscrappy` with its default separable median filter is 1.5 to 1.9 times faster than `dfcosmic` on the CPU, but it produces a different mask (see the Results section).

The runtime of `dfcosmic` depends on the number of cosmic rays in the image (see the Software design section), and the mock image contains only 100. We therefore repeated the measurement on a crowded image: the *HST* image of the Results section, repeated to fill 4000x6500 pixels, in which 3% of the pixels are cosmic rays. There, `dfcosmic` with its C++ median filter takes 34% less time than `astroscrappy` with a true median filter on one thread, 30% less on two and 19% less on eight, and the same time on 16; on the GPU the image takes 0.44 s. Even a moderate gain for each individual frame corresponds to a considerable gain when running the pipeline on several tens of thousands of frames nightly. We reiterate that `dfcosmic` always uses a true median (i.e. `sepmed=False`).


# Software design
`dfcosmic` was designed to be a simple PyTorch implementation of the cosmic ray reduction algorithm initially developed in [@van_dokkum_cosmic-ray_2001]. 
The code was complexified in order to achieve greater reductions in speed. Notably, initial benchmarking revealed the median filter to be the main bottleneck. Therefore, we provide an optional C++ implementation of the median filter for the CPU, which is built from source against the installed version of PyTorch. In addition, three of the five median filters of an iteration are only needed at a small number of pixels: the two that build the fine-structure image are needed at the candidate pixels, and the one used to replace the cosmic rays at the flagged pixels. `dfcosmic` evaluates them at those pixels only, which gives exactly the same result as filtering the whole image and roughly halves the runtime. We chose to use PyTorch instead of a more standard library, such as numpy or scipy, so that we could take advantage of the GPU, if available. As demonstrated in \autoref{fig:comparison}, the GPU implementation is 8.6 times faster than `astroscrappy` with a true median filter on 16 threads, and 43 times faster than on two threads. Although not explored here, the GPU implementation also allows for batch processing which can enable further speedup. 

In order to ensure the fidelity of the functions run internally, we wrote custom torch implementations of the following: `block_replicate_torch`, `convolve`, `median_filter_torch`, `sigma_clip_pytorch`. The growing of the cosmic ray mask (dilation) is implemented as a convolution. If the C++ median filter has not been built, the code uses the torch implementation of the median filter, which gives identical results; we note that this leads to a worse performance on the CPU as compared to the C++ implementation.


# Research impact statement
`dfcosmic` is a new implementation of the well-known L.A.Cosmic algorithm developed by [@van_dokkum_cosmic-ray_2001]. `dfcosmic` is integrated into the nightly reduction pipeline for the partially-constructed MOTHRA. The adoption of this algorithm has considerable impact on the speed of reductions. During a typical night, a single MOTHRA array takes approximately 1200 raw frames (this is the low end); note that the final version of MOTHRA will have 30 arrays operating simultaneously. The pipeline uses 2 threads per frame. In the benchmark above, on our test machine and with 2 threads, cleaning a frame takes 12.2 to 14.2 seconds with `astroscrappy(sepmed=False)`, depending on the image, and 8.5 seconds with `dfcosmic` and its C++ median filter. When run on all the frames taken in an evening, this is equivalent to saving 4,400 to 6,800 seconds (1.2 to 1.9 hours) of computing time for a single array. 


# Methods

## Algorithm

The algorithm follows the methodology described in detail in [@van_dokkum_cosmic-ray_2001]. Below, we outline the main steps:

1. Run laplacian detection
2. Create a noise model
3. Create significance map
4. Compute initial cosmic ray candidates
5. Reject compact, underampled objects (i.e. stars or HII regions)
6. Determine which neighboring pixels to include
7. Replace cosmic rays with median of neighbors

Importantly, we use the classic median filter rather than any optimized version. We overcome the additional computational costs associated with this computation by implementing our methodology in `PyTorch` [@paszke_pytorch_2019] with the median filter optionally running in C++ on the CPU.

## Main parameters
There are several key parameters that a user can set depending on their specific use case:

1. `objlim`: the contrast limit between cosmic rays and underlying objects
2. `sigfrac`: the fractional detection limit for neighboring pixels
3. `sigclip`: the detection limit for cosmic rays

Furthermore, the user can supply the gain and readnoise. If a gain is not supplied, then it will be estimated at each iteration, as in the original implementation; this requires that the sky background has not been subtracted from the image.

# Results

## Example
In order to showcase `dfcosmic`, we apply it, along with `astroscrappy` and `lacosmic` implementations, to the original example from [@van_dokkum_cosmic-ray_2001] of the *HST* WFPC2 image of galaxy cluster MS 1137+67.

![\label{fig:demo} *HST* WFPC2 image of galaxy cluster MS 1137+67. In the top panel, we show the original image, the mask from the original IRAF implementation, and the mask from `dfcosmic`. In the bottom row, we show the mask created by `astroscrappy` with the `sepmed` argument set to True (left) and False (middle) while in the right panel we show the mask from `lacosmic` [@bradley_larrybradleylacosmic_2025].](demos/example_hst.png)

As demonstrated in \autoref{fig:demo}, `dfcosmic` reproduces the mask from the original IRAF implementation more closely than the other Python implementations. Of the 18,921 pixels flagged by IRAF, `dfcosmic` misses 31 and flags 37 others; the intersection over union of the two masks is 0.996. For `astroscrappy` it is 0.957 with a true median filter and 0.912 with its default separable median filter, and for `lacosmic` it is 0.968. By comparison, the other two popular implementation either underestimate (`astroscrappy`) or overestimate (Bradley's `lacosmic`) the size of the CRs in stars. An incorrect masking of CRs in these regions  can have a profound effect on the measured stellar photometries.

# AI usage disclosure
Generative AI was used for the following parts of this project:

1. Claude (Claude.ai and Claude Code) was used to help write the unit tests and to understand the original IRAF implementation.
2. ChatGPT/Codex was used to write the C++ median filter and to make the code more memory efficient.
3. Claude Code was used to help implement the changes requested during the pyOpenSci review: packaging and continuous integration, input validation, the handling of non-finite and unrepairable pixels, the per-iteration gain estimate, the timing benchmark, and documentation.

The original implementation of the algorithm was written by the authors without AI. All code produced by AI was manually inspected for correctness.


# Acknowledgements
We acknowledge the Dragonfly FRO and particularly thank Lisa Sloan for her project management skills.

We thank Robert Vetter, whose review of `dfcosmic` for pyOpenSci led to several corrections and to the evaluation of the median filters only at the pixels where they are needed.

We use the cmcrameri scientific color maps in our demos [@crameri_scientific_2023].

# References
