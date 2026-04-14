# ==========================================================
#   Multi-source Earth Observation Data Cube Generation Script
# ==========================================================

# This script generates data cubes from Sentinel-1, Sentinel-2,
# DEM, and rainfall datasets using the sits package.
# It includes temporal regularization and spectral index computation.

#------------------------------------------
#   Load required packages
#------------------------------------------

#install.packages("torch")
library(torch)
#install.packages("sits")
library(sits)
#install.packages("sf")
library(sf)
#install.packages("gdalcubes")
library(gdalcubes)

#------------------------------------------
#   User-defined parameters (must be updated)
#------------------------------------------

# Sentinel-2 tiles covering the region of interest
tiles <- c("39LTG", "39LUG","38LRL","39LTF","39LUF")

# Working directory path
directory <- "/home/users/h/havinga/MapNorthMadagascar/"
directory <- "path/to/project/"

# Output directory for Sentinel-2 cubes
outputS2 <- paste0(directory,"Data/CubeS2")

# Output directory for Sentinel-2 indices
outputS2ind <- paste0(directory,"Data/CubeS2indices")

# Output directory for Sentinel-1 cubes
outputS1 <- paste0(directory,"Data/CubeS1")

# Output directory for DEM
outputDEM <- paste0(directory,"Data/DEM")

# Output directory for rainfall data
outputRain <- paste0(directory,"Data/RAIN")

# Bands of interest for Sentinel-2 imagery
bandsS2 = c("B01","B02","B03","B04","B05","B06","B12", "B8A", "B11", "CLOUD")

# Bands of interest for Sentinel-1 imagery
bandsS1 = c("VV", "VH")

# Sentinel-1 orbit direction (choose between "ascending" or "descending")
orbit = "descending"

# Time range for image selection
start_date = "2021-01-01"
end_date = "2025-01-01"

# Temporal resolution for the time series (e.g., one image every 4 months)
period = "P4M"

# Spatial resolution in meters
res = 10

# Number of CPU cores used for parallel processing
multicores = 16

#création si besoin des dossiers des cubes
dir.create(outputS2, recursive = TRUE, showWarnings = FALSE)
dir.create(outputS2ind, recursive = TRUE, showWarnings = FALSE)
dir.create(outputS1, recursive = TRUE, showWarnings = FALSE)
dir.create(outputDEM, recursive = TRUE, showWarnings = FALSE)
dir.create(outputRain, recursive = TRUE, showWarnings = FALSE)

#--------------------------------------------------------
#   1. SENTINEL-2 CUBE GENERATION AND REGULARIZATION
#--------------------------------------------------------
# Use sits_list_collections() to check available collections

# All tiles are stored in a single directory to allow joint cube reconstruction

for (i in tiles) {
  
  # Create Sentinel-2 cube over the study area (can take several hours)
  s2_cube <- sits_cube(
    source = "DEAFRICA",
    collection = "SENTINEL-2-L2A",
    tiles = i,
    bands = bandsS2,
    start_date = start_date,
    end_date = end_date)
  print("Sentinel-2 cube created")
  
  # 1.b Regularize the Sentinel-2 cube (temporal and spatial alignment)
  s2_reg <- sits_regularize(
    cube = s2_cube,
    period = period,
    res = res,
    multicores = multicores,
    output_dir = outputS2
  )
  print (paste("Processed tile:", i))
}

#-------------------------------------------------------
#   2. COMPUTE SPECTRAL INDICES FROM SENTINEL-2 BANDS
#--------------------------------------------------------

# Load the regularized Sentinel-2 cube
s2_reg <- sits_cube(
  source      = "DEAFRICA",
  collection  = "SENTINEL-2-L2A",
  tiles = tiles,
  data_dir    = outputS2
)


# NDVI (Normalized Difference Vegetation Index)
s2_reg <- sits_apply(data = s2_reg,
                     NDVI = (B8A - B04) / (B8A + B04),
                     output_dir = outputS2ind
)

plot(s2_reg, band = "NDVI", palette = "RdYlGn")

# EVI (Enhanced Vegetation Index)
s2_reg <- sits_apply(data = s2_reg,
                     EVI = (2.5 * ((B8A - B04) / ((B8A + 6 * B04 - 7.5 * B02) + 1))),
                     output_dir = outputS2ind
)

plot(s2_reg, band = "EVI", palette = "RdYlGn")

# SIPI (Structure Insensitive Pigment Index)
s2_reg <- sits_apply(data = s2_reg,
                     SIPI = (B8A - B01) / (B8A - B04),
                     output_dir = outputS2ind
)

plot(s2_reg, band = "SIPI", palette = "RdYlGn")

# GNDVI (Green NDVI)
s2_reg <- sits_apply(data = s2_reg,
                     GNDVI = ((B8A-B03) / (B8A+B03 )),
                     output_dir = outputS2ind
)

plot(s2_reg, band = "GNDVI", palette = "RdYlGn")

# NDMI (Normalized Difference Moisture Index)
s2_reg <- sits_apply(data = s2_reg,
                     NDMI = ((B8A-B11) / (B8A+B11 )),
                     output_dir = outputS2ind
)

plot(s2_reg, band = "NDMI", palette = "RdYlGn")

# BSI (Bare Soil Index)
s2_reg<- sits_apply(data = s2_reg,
                    BSI = ((B04+B11) -(B8A+B02)) / ((B04+B11) + (B8A+B02)),
                    output_dir = outputS2ind
)

#-------------------------------------------------------
#   3. SENTINEL-1 CUBE GENERATION AND REGULARIZATION
#--------------------------------------------------------

for (i in tiles) {
  # Create Sentinel-1 cube
  s1_cube<-sits_cube(source="DEAFRICA",
                     collection = "SENTINEL-1-RTC",
                     bands = bandsS1,
                     orbit = orbit,
                     tiles = i,
                     start_date = start_date,
                     end_date = end_date
  )
  # Regularize Sentinel-1 cube
  s1_reg <- sits_regularize(
    cube = s1_cube,
    period = period,
    res = res,
    tiles = i,
    multicores = multicores,
    output_dir =  outputS1
  )
  print (i)
  print(paste("Processed tile:", i))
}

#-------------------------------------------------------
#   4. DEM CUBE GENERATION AND REGULARIZATION
#--------------------------------------------------------

# Create DEM cube
cube_dem <- sits_cube(
  source     = "MPC",
  collection = "COP-DEM-GLO-30",
  tiles      = tiles,
  bands      = "ELEVATION"
)

#regularize DEM cube
cube_dem_reg <- sits_regularize(
  cube       = cube_dem,
  res        = res,
  tiles      = tiles,
  output_dir = outputDEM,
  multicores = multicores
)

#-------------------------------------------------------
# 5. RAINFALL CUBE GENERATION AND REGULARIZATION
#--------------------------------------------------------

# Create rainfall cube (CHIRPS dataset)
Rain_Cube <-sits_cube(source="DEAFRICA",
                          collection = "RAINFALL-CHIRPS-MONTHLY",
                          roi = sits_mgrs_to_roi(tiles),
                          start_date = start_date,
                          end_date = end_date
)

# Regularize rainfall cube

# Rainfall data (original resolution ~1 km) are resampled to 10 m
# to match the spatial resolution of other datasets.
# This does not increase the intrinsic spatial accuracy.

Rain_Reg <- sits_regularize(
  cube = Rain_Cube ,
  period = period,
  res = res,
  tiles = tiles,
  multicores = multicores,
  output_dir = outputRain
)

