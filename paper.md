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


# Statement of need
Although several implementations LA Cosmic ([@van_dokkum_cosmic-ray_2001]) exist such as lacosmic ([@bradley_larrybradleylacosmic_2025]) and astroscrappy ([@robitaille_astropyastroscrappy_2025]), these implementations either deviate from the original algorithm in order to achieve computational gains or do not run fast enough for practical usage. In particular, the data reduction pipeline for the MOTHRA instrument requires rapid cosmic ray identification and removal for tens of thousands of images every night using only a single core (2 threads) per image. Importantly, experiments on the preliminary data have demonstrated that it is crucial to use the original implementation (notably a true median filter rather than a sliding or separable median filter) in order to capture all the cosmic rays without accidentally removing bright stars.


# Methods

## Algorithm

The algorithm follows the methodology described in detail in [@van_dokkum_cosmic-ray_2001]. Below, we outline the main steps:

1. Run laplacian detection
2. Create a noise model
3. Create significance map
4. Compute initial cosmic ray candidates
5. Reject objects (i.e. stars or HII regions)
6. Determine which neighboring pixels to include
7. Replace confirmed cosmic rays with median of neighbors

Importantly, we use the classic median filter rather than any optimized version. We overcome the additional computational costs associated with this computation by implementing our methodology in `PyTorch` [@paszke_pytorch_2019].

## Parameters
There are several key parameters that a user can set depending on their specific use case:

1. `objlim`: the contrast limit between cosmic rays and underlying objects
2. `sigfrac`: the fractional detection limit for neighboring pixels
3. `sigclip`: the detection limit for cosmic rays

Additionally, the user can supply the gain and readnoise. If a gain is not supplied, then it will be estimated at each iteration.

# Results

## Example
In order to showcase `dfcosmic`, we construct an example image that is 100x100 pixels containing two fake elliptical galaxies and 10 stars with Gaussian noise. We then add 10 cosmic rays. We subsequently run `dfcosmic` with the default parameters. In the image below (\autoref{fig:demo}), we show the clean mock image, the cosmic ray mask, the dirty mock image (clean mock image + cosmic rays), the `dfcosmic` image, and the `dfcosmic` mask. We see that the algorithm correctly detects all cosmic rays in the image; additionally, it incorrectly marks two noisy pixels at the boundary as cosmic rays. This is a known complication with the algorithm. However, these noisy pixels are replaced with a median of their neighboring pixels and thus do not change the noise properties of the image.

![\label{fig:demo} Example of `dfcosmic` on a mock image containing fake elliptical galaxies, stars, and cosmic rays.](demo/example.png)

## Speed Testing
An important aspect of `dfcosmic` is that it reduces computation time while using the classic median filter. In order to test this, we run the following codes under the following conditions on the mock data used for testing by [astroscrappy]. We run the following options:
 
 - `dfcosmic` on CPU
 - `dfcosmic` on GPU
 - `dfcosmic` with `scipy`
 - `astroscrappy` with `sepmed=True`
 - `astroscrappy` with `sepmed=False`
 - `lacosmic`

Additionally, we run each option with 1, 2, 4, and 8 threads. 

# Acknowledgements
We acknowledge the Dragonfly FRO and particularly thank Lisa Sloan for her project management skills.

We use the cmcrameri scientific color maps in our demos [@crameri_scientific_2023].

# References
