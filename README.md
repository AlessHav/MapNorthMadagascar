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

---

## Notes

All Sentinel tiles are stored in a single directory to enable reconstruction of multi-tile data cubes.
Rainfall data (original coarse resolution) are resampled to 10 m to match other datasets. This does not increase spatial accuracy.
Processing time can be several hours depending on the number of tiles and computing resources.

---

## Output

The script generates:

Sentinel-2 cubes
Spectral indices
Sentinel-1 cubes
DEM cube
Rainfall cube

Each dataset is saved in its respective output directory.

---

## Author

Alessandra Havinga 
alessandra.havinga@geneve.ch

Install them using:

```r
install.packages(c("sits", "torch", "sf", "gdalcubes"))
