# =====================================================
# GEE → SITS PIPELINE (DOCUMENTED VERSION)
# =====================================================
# This script converts probability maps exported from
# Google Earth Engine into SITS-compatible data cubes.
#
# Main steps:
# 1. Mosaic spatially split GEE exports
# 2. Align to Sentinel-2 grid
# 3. Reorder class probabilities
# 4. Calibrate probabilities (sharpening + normalization)
# 5. Export optimized GeoTIFF
# =====================================================

library(terra)     # raster processing
library(stringr)   # string parsing

# =====================================================
# 1. CONFIGURATION
# =====================================================

# Central configuration list
# Ensures reproducibility and easy parameter tuning
config <- list(
  paths = list(
    #gee_dir     = "data/raw",           # GEE exported tiles
    gee_dir     = "/home/users/h/havinga/MapNorthMadagascar/Data/Roi1ProbaGEENew",
    #s2_dir      = "data/s2",            # Sentinel-2 reference grid
    s2_dir     = "/home/users/h/havinga/CarteNMada/S2"
  ),
  # Temporal coverage of the exported probability cubes
  temporal = list(
    start_date = "2021-01-01",
    end_date   = "2024-09-01"
  ),
  
  calibration = list(
    alpha = 1.7,   # Empirical sharpening parameter
    scale = 10000  # Required scaling for SITS compatibility
  ),
  
  # Expected class order (must match SITS model)
  class_order = c(
    "1011","1012","102","103","104","105",
    "1131","1132","1133","211","221","231","232",
    "311","312","313","321","322"
  )
)
# # Build derived output directories
config$paths$mosaic_dir <- file.path(config$paths$gee_dir, "MosaicGEETile")
config$paths$aligned_dir <- file.path(config$paths$gee_dir, "AlignedGEETile")
config$paths$output_dir <- file.path(config$paths$gee_dir, "ProcessedGEETile")

# Create output directories if they do not exist
dir.create(config$paths$mosaic_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(config$paths$aligned_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(config$paths$output_dir, recursive = TRUE, showWarnings = FALSE)

# =====================================================
# 2. UTILITIES
# =====================================================

# Extract Sentinel-2 tile ID (MGRS format) from filename
# Example: "T39LUG" or "39LUG"
extract_tile <- function(x) {
  str_extract(basename(x), "[0-9]{2}[A-Z]{3}")
}

# Build SITS-compliant filename
build_filename <- function(tile, start_date, end_date) {
  paste0(
    "SENTINEL-2_MSI_",
    tile, "_",
    start_date, "_",
    end_date,
    "_probs_v1.tif"
  )
}

# Validate that all expected classes are present
# Prevents silent errors during layer reordering
check_classes <- function(r, class_order) {
  
  # Remove prefix if present (e.g., "prob_")
  names(r) <- gsub("^prob_", "", names(r))
  
  # Identify missing classes
  missing <- setdiff(class_order, names(r))
  
  # Stop execution if any class is missing
  if (length(missing) > 0) {
    stop("Missing classes: ", paste(missing, collapse = ", "))
  }
  
  return(TRUE)
}

# =====================================================
# 3. MOSAIC GEE EXPORTS
# =====================================================

# Merge multiple raster chunks belonging to the same tile
# GEE often exports large regions as spatial fragments
mosaic_tiles <- function(files, out_dir) {
  
  # Group files by tile ID
  split_files <- split(files, sapply(files, extract_tile))
  outputs <- list()
  
  for (tile in names(split_files)) {
    
    message("Mosaicking tile: ", tile)
    
    # Load all fragments for the tile
    rasters <- lapply(split_files[[tile]], rast)
    
    # Merge into a single raster
    mos <- do.call(mosaic, rasters)
    
    # Save mosaicked raster
    out_file <- file.path(out_dir, paste0(tile, "_mosaic.tif"))
    writeRaster(mos, out_file, overwrite = TRUE)
    
    outputs[[tile]] <- out_file
  }
  
  return(outputs)
}

# =====================================================
# 4. ALIGNMENT TO SENTINEL-2 GRID
# =====================================================

# Reproject raster to match Sentinel-2 geometry exactly
# Ensures compatibility with SITS data cubes
align_tile <- function(mosaic_file, s2_dir, class_order, out_dir) {
  
  tile <- extract_tile(mosaic_file)
  
  # Find corresponding Sentinel-2 reference tile
  ref_file <- list.files(s2_dir, pattern = tile, full.names = TRUE)[1]
  
  if (is.na(ref_file)) {
    stop("No S2 reference for tile: ", tile)
  }
  
  # Load rasters
  r <- rast(mosaic_file)
  ref <- rast(ref_file)
  
  message("Aligning tile: ", tile)
  
  # Reproject using nearest neighbor
  # (preserves class probabilities without interpolation)
  r2 <- project(
    r, ref,
    method = "near",
    filename = tempfile(fileext = ".tif"),  # avoids RAM overflow
    overwrite = TRUE
  )
  
  # Ensure class completeness
  check_classes(r2, class_order)
  
  # Reorder layers to match expected class order
  names(r2) <- gsub("^prob_", "", names(r2))
  r2 <- r2[[class_order]]
  names(r2) <- paste0("prob_", class_order)
  
  # Save aligned raster
  out_file <- file.path(out_dir, paste0(tile, "_aligned.tif"))
  
  writeRaster(
    r2,
    out_file,
    overwrite = TRUE,
    filetype = "GTiff",
    gdal = c(
      "COMPRESS=LZW",  # lossless compression
      "TILED=YES"      # optimized for large rasters
    )
  )
  
  return(out_file)
}

# =====================================================
# 5. PROBABILITY CALIBRATION
# =====================================================

# Apply probability sharpening and normalization
# Ensures:
# - probabilities sum to 1 per pixel
# - improved class separability
# - compatibility with SITS scaling requirements

calibrate_tile <- function(file, alpha, scale, out_dir) {
  
  r <- rast(file)
  
  if (alpha > 1)
    r <- r ^ alpha
  
  r_sum <- app(
    r,
    sum,
    na.rm = TRUE,
    filename = tempfile(fileext = ".tif"),
    overwrite = TRUE
  )
  
  r_sum[r_sum == 0] <- NA
  
  r_scaled <- round(
    clamp(r / r_sum, 0, 1) * scale
  )
  
  out_file <- file.path(out_dir, build_filename(
    extract_tile(file),
    config$temporal$start_date,
    config$temporal$end_date
  ))
  
  writeRaster(
    r_scaled,
    out_file,
    overwrite = TRUE,
    datatype = "INT2U",
    gdal = c(
      "COMPRESS=DEFLATE",
      "PREDICTOR=2",
      "TILED=YES"
    )
  )
  rm(r, r_sum, r_scaled)
  gc()
  
  out_file
}
# =====================================================
# 6. RUN PIPELINE
# =====================================================

message("=== START PIPELINE ===")

# List all GEE exported files
gee_files <- list.files(config$paths$gee_dir, "\\.tif$", full.names = TRUE)

# Stop if no input data is found
if (length(gee_files) == 0) {
  stop("No GEE files found.")
}

# Step 1 — Mosaic spatial fragments
mosaics <- mosaic_tiles(gee_files, config$paths$mosaic_dir)

# Optional: save intermediate outputs to avoid rerunning
# computationally intensive steps.
saveRDS(mosaics, file = "Data/Roi1ProbaGEENew/mosaics.rds")
mosaics <- readRDS("Data/Roi1ProbaGEENew/mosaics.rds")

# Step 2 — Align to Sentinel-2 grid
aligned_files <- lapply(
  mosaics,
  align_tile,
  s2_dir = config$paths$s2_dir,
  class_order = config$class_order,
  out_dir = config$paths$aligned_dir
)

# Optional: save intermediate outputs to avoid rerunning
# computationally intensive steps.
saveRDS(aligned_files, file = "Data/Roi1ProbaGEENew/aligned_files2.rds")
aligned_files<-readRDS("Data/Roi1ProbaGEENew/aligned_files2.rds")

# Step 3 — Calibrate probabilities
final_outputs <- lapply(
  aligned_files,
  calibrate_tile,
  alpha = config$calibration$alpha,
  scale = config$calibration$scale,
  out_dir = config$paths$output_dir
)

message("=== probability cubes generated ===")