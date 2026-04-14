# Multi-source Earth Observation Data Cube Generation

## Overview
This repository provides an R script to generate harmonized Earth Observation data cubes using the `sits` package.

The workflow integrates:
- Sentinel-2 (optical imagery)
- Sentinel-1 (SAR imagery)
- DEM (elevation)
- Rainfall (CHIRPS)

All datasets are temporally regularized and spatially aligned.

---

## Features

- Multi-source data cube generation
- Temporal regularization (4-month intervals)
- Spectral index computation (NDVI, EVI, SIPI, GNDVI, NDMI & BSI)
- Harmonized spatial resolution (10 m)
- Parallel processing support

---

## Requirements

R packages:
- sits
- torch
- sf
- gdalcubes

Install them using:

```r
install.packages(c("sits", "torch", "sf", "gdalcubes"))
