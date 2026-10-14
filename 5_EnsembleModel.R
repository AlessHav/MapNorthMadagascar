# =====================================================
# 5. ENSEMBLE CLASSIFICATION PIPELINE
# =====================================================
# This script combines the probability cubes produced by
# different classification approaches into a single
# ensemble prediction.

# The workflow also includes the generation of validation
# samples, the assessment of classification accuracy,
# and optional class aggregation for thematic analyses.

# Main steps:
# 0. Load input data and parameters
# 1. Define the ensemble prediction function
# 2. Load the probability cubes
# 3. Generate ensemble classification maps
# 4. Crop classification maps to the Region of Interest
# 5. Generate validation samples
# 6. Prepare the validation dataset
# 7. Generate classification mosaics
# 8. Accuracy assessment
# =====================================================
#------------------------------------------
# Load the required packages
#------------------------------------------

#install.packages("torch")
library(torch)

#install.packages("sits")
library(sits)

#install.packages("sf")
library(sf)

#install.packages("dplyr")
library(dplyr)
#--------------------------------------------------------------------
# 0. Load input data
#--------------------------------------------------------------------
# Define the main project directories.
# Update these paths to match your local project structure.
directoryData <- "/home/users/h/havinga/MapNorthMadagascar/Data"
directory <- "/home/users/h/havinga/CarteNMada/NEW_ROI1"
RDSdirect <- "/home/users/h/havinga/MapNorthMadagascar/Data/SaveRDS2026/EnsembleModel/"

# Load the Region of Interest (ROI).
# Update the shapefile path if using a different study area.
roi <- st_read(paste0(directoryData, "/ROI/RoiReady1.shp"))

plot(roi)


# Computational parameters
# Adjust these parameters according to the computational
# resources available on your computer or server.

multicores <- 4
memsize <- 64
#---------------------------------------------------------------
#Define the land-cover class labels
#---------------------------------------------------------------

# Class labels used when importing probability cubes.
# The order must match the probability bands generated
# by the different classification models.
label <- c(
  "1011","1012","102","103","104","105",
  "1131","1132","1133","211","221","231","232",
  "311","312","313","321","322"
)

names(label) <- c(1:18)

#--------------------------------------------------------------
# 1. Ensemble prediction function
#-------------------------------------------------------------- 

# Combine multiple probability cubes using a weighted average,
# apply spatial smoothing, and generate the final thematic map.
#
# Arguments:
#   cubes      : list of probability cubes
#   version    : output version name
#   weights    : weight assigned to each probability cube
#   directory  : output directory


EnsembleModel<-function (cubes, version, weights, directory){
  
  # Combine the probability cubes using a weighted average
  patchwork<- sits_combine_predictions(
    cubes = cubes,
    type = "average",
    version = version,
    output_dir =paste0(directory,"/EnsembleModel"),
    weights = weights,
    memsize = memsize,
    multicores = multicores
  )
  # Apply Bayesian spatial smoothing.
  # The smoothing parameters were previously estimated
  # from the logit variance analysis (see Script 2_MapCreation).
  patchworksmoothparam <- sits_smooth(
    patchwork,
    output_dir = paste0(directory,"/EnsembleModel"),
    version = paste0(version,"sm95"),
    #optimize smoothness parameter from the 2_MapCreation.R code
    smoothness = c(
      "1011" = 0.07,
      "1012" = 0.03,
      "102" = 0.11,
      "103" = 0.180,
      "104" = 0.03,
      "105" = 0.08,
      "1131" = 0.1,
      "1132" = 0.25,
      "1133" = 0.22,
      "211" = 0.06,
      "221" = 0.19,
      "311" = 0.04,
      "312" = 0.01,
      "313"= 0.1,
      "321" = 0.04,
      "322" = 0.1,
      "231" = 0.27,
      "232" = 0.12),
    multicores = multicores,  
    memsize = memsize
  )
  # Convert the smoothed probability cube
  # into the final land-cover classification map.
  labels <- sits_label_classification(
    cube = patchworksmoothparam,
    multicores = multicores,
    memsize = memsize,
    output_dir = paste0(directory,"/EnsembleModel"),
    version = paste0(version,"sm95"),
    threshold = 0.1
  )
  
  # Create the output directory if it does not exist.
  rds_output_dir <- file.path(
    directoryData,
    "SaveRDS2026",
    "EnsembleModel"
  )
  
  dir.create(
    rds_output_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )
  # Save the classified map
  saveRDS(labels,file = file.path(rds_output_dir, paste0(version, "sm95.rds")))
  return(labels)
}

#--------------------------------------------------------------
# 2. Load the probability cubes
#--------------------------------------------------------------
# Set data_dir to the directory containing the input probability tiles.

# Load the probability cube generated in
# Google Earth Engine (script 4)
GEE_probs <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = "Data/Roi1ProbaGEENew/ProcessedGEETile",
  bands = "probs",
  labels = label,
  roi=roi, 
  parse_info = c("satellite", "sensor", "tile", 
                 "start_date", "end_date", "band", "version"))


# Load the TempCNN probability cube.
TempCNN_probs <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData, "/Map2026/TempCNN"),
  bands = "bayes",
  labels = label,
  roi=roi, 
  parse_info = c("satellite", "sensor", "tile", 
                 "start_date", "end_date", "band", "version"))

# Load the MLP probability cube.
MLP_probs <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData, "/Map2026"),
  bands = "bayes",
  labels = label,
  roi=roi,
  version = "mlp-smooth95-roi1",
  parse_info = c("satellite", "sensor", "tile", 
                 "start_date", "end_date", "band", "version"))

# Load the XGB probability cube.
XGB_probs <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData, "/Map2026"),
  bands = "bayes",
  labels = label,
  roi=roi,
  version= "2-95%-xgb-balanced-roi1",
  parse_info = c("satellite", "sensor", "tile", 
                 "start_date", "end_date", "band", "version"))

# Load the RF probability cube.
RF5_probs <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData, "/Map2026"),
  bands = "bayes",
  labels = label,
  roi=roi,
  version = "5-95%-rf-balanced-roi1",
  parse_info = c("satellite", "sensor", "tile", 
                 "start_date", "end_date", "band", "version"))

#----------------------------------------------------------
# 3. Generate ensemble classification maps
#----------------------------------------------------------

# The following examples illustrate different
# ensemble weighting strategies.

# The weights define the relative contribution of each
# classification model to the final ensemble prediction.

# Example: 34% Google Earth Engine, 33% TempCNN,
# and 33% Random Forest.

labels <- EnsembleModel(
  cubes = list(GEE_probs, TempCNN_probs, RF5_probs), 
  version = "A-34GEE-33Temp-33RF5", 
  weights= c(0.34 , 0.33, 0.33),
  directory = paste0(directoryData, "/Map2026")
)

# Example: 50% Google Earth Engine,
# 25% TempCNN, and 25% Random Forest.

labels2 <- EnsembleModel(
  cubes = list(GEE_probs, TempCNN_probs, RF5_probs), 
  version = "A-50GEE-25Temp-25RF5", 
  weights= c(0.50 , 0.25, 0.25),
  directory = paste0(directoryData, "/Map2026")
)

# Example: Equal contribution of
# Google Earth Engine and TempCNN.
labels3 <- EnsembleModel(
  cubes = list(GEE_probs, TempCNN_probs), 
  version = "A-50GEE-50Temp95", 
  weights= c(0.50 , 0.50),
  directory = paste0(directoryData, "/Map2026")
)

# Example: 75% Google Earth Engine,
# 25% TempCNN.
labels4 <- EnsembleModel(
  cubes = list(GEE_probs, TempCNN_probs), 
  version = "A-75GEE-25Temp95", 
  weights= c(0.75 , 0.25),
  directory = paste0(directoryData, "/Map2026")
)

# Example: 25% Google Earth Engine,
# 75% TempCNN.
labels5 <- EnsembleModel(
  cubes = list(GEE_probs, TempCNN_probs), 
  version = "A-25GEE-75Temp95", 
  weights= c(0.25 , 0.75),
  directory = paste0(directoryData, "/Map2026")
)

# Example: 63% Google Earth Engine,
# 37% TempCNN.
labels6 <- EnsembleModel(
  cubes = list(GEE_probs, TempCNN_probs), 
  version = "A-63GEE-37Temp95", 
  weights= c(0.63 , 0.37),
  directory = paste0(directoryData, "/Map2026")
)

# Example: 87% Google Earth Engine,
# 13% TempCNN.
labels7 <- EnsembleModel(
  cubes = list(GEE_probs, TempCNN_probs), 
  version = "A-87GEE-13Temp95", 
  weights= c(0.87 , 0.13),
  directory = paste0(directoryData, "/Map2026")
)

# Example: 75% Google Earth Engine,
# 25% Random Forest.
labels8 <- EnsembleModel(
  cubes = list(GEE_probs, RF5_probs), 
  version = "A-75GEE-25RF5", 
  weights= c(0.75 , 0.25),
  directory = paste0(directoryData, "/Map2026")
)


labels9 <- EnsembleModel(
  cubes = list(GEE_probs, MLP_probs), 
  version = "A-75GEE-25MLP95", 
  weights= c(0.75 , 0.25),
  directory = paste0(directoryData, "/Map2026")
)

labels10 <- EnsembleModel(
  cubes = list(GEE_probs, XGB_probs), 
  version = "A-75GEE-25XBG95", 
  weights= c(0.75 , 0.25),
  directory = paste0(directoryData, "/Map2026")
)


labels11 <- EnsembleModel(
  cubes = list(GEE_probs, MLP_probs), 
  version = "A-87GEE-13MLP95", 
  weights= c(0.87 , 0.13),
  directory = paste0(directoryData, "/Map2026")
)

labels12 <- EnsembleModel(
  cubes = list(GEE_probs, MLP_probs), 
  version = "A-63GEE-37MLP95", 
  weights= c(0.63 , 0.37),
  directory = paste0(directoryData, "/Map2026")
)
labels13 <- EnsembleModel(
  cubes = list(GEE_probs, XGB_probs), 
  version = "A-87GEE-13XBG95", 
  weights= c(0.87 , 0.13),
  directory = paste0(directoryData, "/Map2026")
)

labels14 <- EnsembleModel(
  cubes = list(GEE_probs, XGB_probs), 
  version = "A-63GEE-37XBG95", 
  weights= c(0.63 , 0.37),
  directory = paste0(directoryData, "/Map2026")
)
labels15 <- EnsembleModel(
  cubes = list(GEE_probs, RF5_probs), 
  version = "A-87GEE-13RF5", 
  weights= c(0.87 , 0.13),
  directory = paste0(directoryData, "/Map2026")
)
labels16 <- EnsembleModel(
  cubes = list(GEE_probs, RF5_probs), 
  version = "A-63GEE-37RF5", 
  weights= c(0.63 , 0.37),
  directory = paste0(directoryData, "/Map2026")
)

#----------------------------------------------------------
# 4. Crop the classification maps to the Region of Interest
#----------------------------------------------------------

# Apply the Region of Interest (ROI) mask to all
# classified raster maps. This step removes pixels
# located outside the study area (mainly ocean pixels)
# before generating the validation samples.

#----------------------------------------------------------
# Crop and mask raster maps using the Region of Interest.
#
# Arguments:
#   path      : directory containing the input rasters
#   pattern   : filename pattern used to select rasters
#   outputdir : output directory
#   roi       : Region of Interest (SpatVector)
#
# Returns:
#   Cropped raster maps saved to the output directory.
#----------------------------------------------------------

crop_rasters_roi <- function(path,
                             pattern,
                             outputdir,
                             roi) {
  
  # Check that the input directory exists
  if (!dir.exists(path)) {
    stop("The input directory does not exist: ", path)
  }
  
  # Create the output directory if necessary
  if (!dir.exists(outputdir)) {
    dir.create(outputdir, recursive = TRUE)
    message("Output directory created: ", outputdir)
  }
  
  #  Search for raster files matching the specified pattern
  rasters <- list.files(
    path = path,
    pattern = pattern,
    full.names = TRUE,
    ignore.case = TRUE
  )
  
  cat(length(rasters), "rasters found\n\n")
  
  if (length(rasters) == 0) {
    warning("No raster files found.")
    return(invisible(NULL))
  }
  
  # Process each raster independently
  for (f in rasters) {
    
    cat("Processing :", basename(f), "\n")
    
    # Load the raster
    r <- terra::rast(f)
    
    # Reproject the ROI to match the raster coordinate reference system
    roi_local <- terra::project(roi, terra::crs(r))
    
    # Check whether the raster intersects the ROI
    test_crop <- try(
      terra::crop(r, roi_local),
      silent = TRUE
    )
    
    if (inherits(test_crop, "try-error")) {
      cat("  -> no intersection with the ROI\n\n")
      next
    }
    
    # Crop the raster to the ROI extent
    r_crop <- terra::crop(r, roi_local)
    # Apply the ROI as a spatial mask
    r_crop <- terra::mask(r_crop, roi_local)
    
    # Define the output filename
    out_file <- file.path(
      outputdir,
      basename(f)
    )
    
    # Save the cropped raster
    terra::writeRaster(
      r_crop,
      out_file,
      overwrite = TRUE
    )
    
    cat("  -> saved :", basename(out_file), "\n\n")
  }
  
  invisible(TRUE)
}

roi <- terra::vect(roi)
#----------------------------------------------------------
# Apply the cropping function to each classification map
#----------------------------------------------------------

# TempCNN
crop_rasters_roi(
  path = file.path(directoryData, "Map2026"),
  pattern = "class_2-tcnn-map-95-newbal.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/TempCNN_2"),
  roi = roi
)
#RandomForest
crop_rasters_roi(
  path = file.path(directoryData, "Map2026"),
  pattern = "class_5-95%-rf-roi1.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/RF_5"),
  roi = roi
)

#bisGee34-TempCNN33-RF33
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-34gee-33temp-33rf5sm95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee34-TempCNN33-RF533"),
  roi = roi
)
#Gee50-TempCNN25-RF25
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-50gee-25temp-25rf5sm95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee50-TempCNN25-RF525"),
  roi = roi
)

#Gee25-TempCNN75
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-25gee-75temp95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee25-TempCNN75"),
  roi = roi
)
#Gee50-TempCNN50
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-50gee-50temp95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee50-TempCNN50"),
  roi = roi
)
#Gee75-TempCNN25
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-75gee-25temp95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee75-TempCNN25"),
  roi = roi
)
#Gee87-TempCNN13
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-87gee-13temp95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee87-TempCNN13"),
  roi = roi
)

#Gee63-TempCNN37
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-63gee-37temp95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee63-TempCNN37"),
  roi = roi
)

#Gee75-RF25
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-75gee-25rf5sm95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee75-RF525"),
  roi = roi
)

#Gee75-MLP25
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-75gee-25mlp95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee75-MLP25"),
  roi = roi
)
#Gee87-MLP13
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-87gee-13mlp95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee87-MLP13"),
  roi = roi
)
#Gee63-MLP37
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-63gee-37mlp95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee63-MLP37"),
  roi = roi
)

#Gee75-XGB25
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-75gee-25xbg95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee75-XGB25"),
  roi = roi
)
#Gee87-XGB13
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-87gee-13xbg95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee87-XGB13"),
  roi = roi
)
#Gee63-XGB37
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-63gee-37xbg95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee63-XGB37"),
  roi = roi
)

#MLP
crop_rasters_roi(
  path = file.path(directoryData, "Map2026"),
  pattern = "class_2-95%-mlp-clean-roi1.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee75-MLP25"),
  roi = roi
)

#XGB
crop_rasters_roi(
  path = file.path(directoryData, "Map2026"),
  pattern = "class_2-95%-xbg-roi1.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee75-XGB25"),
  roi = roi
)

#Gee75-RF525
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-75gee-25rf5sm95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee75-FR525"),
  roi = roi
)
#Gee87-RF513
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-87gee-13rf5sm95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee87-FR513"),
  roi = roi
)
#Gee63-RF537
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-63gee-37rf5sm95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee63-FR537"),
  roi = roi
)
#RF5
crop_rasters_roi(
  path = file.path(directoryData, "Map2026"),
  pattern = "class_5-95%-rf-roi1.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/RF5"),
  roi = roi
)

#----------------------------------------------------------
# 5. Generate the validation samples
#----------------------------------------------------------

# Load the cropped classification cube selected
# for designing the validation sampling scheme.
cropmap <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData, "/Map2026/Crop_Roi1/Gee75-TempCNN25"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25temp95sm95"
)

# Define the stratified sampling design
# for the validation dataset
sampling_design<- sits_sampling_design(
  cube = cropmap,
  expected_ua = 0.8,
  alloc_options = c(100,50, 80,90,70), 
  std_err = 0.01,
  rare_class_prop = 0.06
)

sampling_design
# Save the sampling design
saveRDS(sampling_design, file = paste0(directoryData, "/SaveRDS2026/sampling_design_cropmap.rds"))

# Generate the validation samples
set.seed(1234)
ro_samples_sf <- sits_stratified_sampling(
  cube = cropmap,
  sampling_design = sampling_design,
  alloc = "alloc_70",
  multicores = multicores
)

# Export the validation samples as a shapefile
# for manual interpretation and quality control.
sf::st_write(ro_samples_sf, 
             file.path(directoryData,"TrainingSamples/ValidSamplestoCheck/valid_samples_tocheck.shp"), 
             append = FALSE
)


#----------------------------------------------------------
# 6. Prepare the validation dataset
#----------------------------------------------------------

# The validation samples were manually checked
# and relabelled in ArcGIS before being re-imported.
# Read the shapefile

valid_samples <- sf::st_read(
  paste0(directoryData,"/Data/validSamplesReady/ValidSamplesReady.shp"),
  quiet = TRUE
)

summary(valid_samples)

#----------------------------------------------------------
# 7. Accuracy assessment
#----------------------------------------------------------

# Load & create mosaic of the classified maps to be evaluated.

# TempCNN classification 
TempCNN <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/TempCNN_2"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "2-tcnn-map-95-newbal"
)

sits_mosaic(
  TempCNN,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "TempCNN",
  progress = TRUE
)


GEE <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee-sits"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "geesm95"
)

sits_mosaic(
  GEE,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "GEE",
  progress = TRUE
)

Gee34 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee34-TempCNN33-RF533"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-34gee-33temp-33rf5sm95"
)
sits_mosaic(
  Gee34,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-34gee-33temp-33rf5sm95",
  progress = TRUE
)

Gee50 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee50-TempCNN25-RF525"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-50gee-25temp-25rf5sm95"
)
sits_mosaic(
  Gee50,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-50gee-25temp-25rf5sm95",
  progress = TRUE
)

Gee25TempCNN75 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee25-TempCNN75"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-25gee-75temp95sm95"
)
sits_mosaic(
  Gee25TempCNN75,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-25gee-75temp95sm95",
  progress = TRUE
)

Gee50TempCNN50 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee50-TempCNN50"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-50gee-50temp95sm95"
)
sits_mosaic(
  Gee50TempCNN50,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-50gee-50temp95sm95",
  progress = TRUE
)

Gee75TempCNN25 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee75-TempCNN25"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25temp95sm95"
)
sits_mosaic(
  Gee75TempCNN25,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-75gee-25temp95sm95",
  progress = TRUE
)

Gee87TempCNN13 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee87-TempCNN13"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-87gee-13temp95sm95"
)
sits_mosaic(
  Gee87TempCNN13,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-87gee-13temp95sm95",
  progress = TRUE
)

Gee63TempCNN37 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee63-TempCNN37"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-63gee-37temp95sm95"
)
sits_mosaic(
  Gee63TempCNN37,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-63gee-37temp95sm95",
  progress = TRUE
)

Gee75RF25 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee75-RF525"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25rf5sm95"
)
sits_mosaic(
  Gee75RF25,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-75gee-25rf5sm95",
  progress = TRUE
)

Gee75MLP25 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee75-MLP25"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25mlp95sm95"
)
sits_mosaic(
  Gee75MLP25,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-75gee-25mlp95sm95",
  progress = TRUE
)
Gee87MLP13 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee87-MLP13"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-87gee-13mlp95sm95"
)
sits_mosaic(
  Gee87MLP13,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-87gee-13mlp95sm95",
  progress = TRUE
)
Gee63MLP37 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee63-MLP37"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-63gee-37mlp95sm95"
)
sits_mosaic(
  Gee63MLP37,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-63gee-37mlp95sm95",
  progress = TRUE
)

Gee75XGB25 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee75-XGB25"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25xbg95sm95"
)
sits_mosaic(
  Gee75XGB25,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-75gee-25xbg95sm95",
  progress = TRUE
)
Gee87XGB13 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee87-XGB13"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-87gee-13xbg95sm95"
)
sits_mosaic(
  Gee87XGB13,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-87gee-13xbg95sm95",
  progress = TRUE
)
Gee63XGB37 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee63-XGB37"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-63gee-37xbg95sm95"
)
sits_mosaic(
  Gee63XGB37,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-63gee-37xbg95sm95",
  progress = TRUE
)

MLP <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee75-MLP25"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "2-95%-mlp-clean-roi1"
)
sits_mosaic(
  MLP,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "2-95%-mlp-clean-roi1",
  progress = TRUE
)

XGB <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee75-XGB25"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "2-95%-xbg-roi1"
)
sits_mosaic(
  XGB,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "2-95%-xbg-roi1",
  progress = TRUE
)

#Gee75-RF525
GEE75RF525 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee75-FR525"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25rf5sm95"
)
sits_mosaic(
  GEE75RF525,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-75gee-25rf5sm95",
  progress = TRUE
)
#Gee87-RF513
GEE87RF513 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee87-FR513"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-87gee-13rf5sm95"
)
sits_mosaic(
  GEE87RF513,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-87gee-13rf5sm95",
  progress = TRUE
)
#Gee63-RF537
GEE63RF537 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee63-FR537"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-63gee-37rf5sm95"
)
sits_mosaic(
  GEE63RF537,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-63gee-37rf5sm95",
  progress = TRUE
)

RF5 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/RF5"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "5-95%-rf-roi1"
)
sits_mosaic(
  RF5,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "5-95%-rf-roi1",
  progress = TRUE
)

Gee34RF5 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee34-TempCNN33-RF533"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-34gee-33temp-33rf5sm95"
)
sits_mosaic(
  Gee34RF5,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-34gee-33temp-33rf5sm95",
  progress = TRUE
)

Gee50RF5 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee50-TempCNN25-RF525"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-50gee-25temp-25rf5sm95"
)
sits_mosaic(
  Gee50RF5,
  crs = "EPSG:32739",
  multicores = multicores,
  output_dir =  paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic"),
  res = NULL,
  version = "a-50gee-25temp-25rf5sm95",
  progress = TRUE
)


# TempCNN classification 
mosaic_dir <- paste0(directoryData,"/Map2026/Crop_Roi1/Mosaic")

TempCNN <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "tempcnn"
)

RF5 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "5-95%-rf-roi1"
)

GEE <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "gee"
)

XGB <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "2-95%-xbg-roi1"
)

MLP <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "2-95%-mlp-clean-roi1"
)

GEE75RF525 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25rf5sm95"
)
GEE63RF537 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-63gee-37rf5sm95"
)
GEE87RF513 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-87gee-13rf5sm95"
)

GEE75XGB25 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25xbg95sm95"
)
GEE87XGB13 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-87gee-13xbg95sm95"
)
GEE63XGB37 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-63gee-37xbg95sm95"
)

GEE75MLP25 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25mlp95sm95"
)
GEE63MLP37 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-63gee-37mlp95sm95"
)
GEE87MLP13 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-87gee-13mlp95sm95"
)

GEE87Temp13 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-87gee-13temp95sm95"
)

GEE75Temp25 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25temp95sm95"
)
GEE63Temp37 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-63gee-37temp95sm95"
)

GEE50Temp50 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-50gee-50temp95sm95"
)

GEE25Temp75 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-25gee-75temp95sm95"
)

GEE34Temp33RF533 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-34gee-33temp-33rf5sm95"
)
GEE50Temp25RF525 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = mosaic_dir,
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-50gee-25temp-25rf5sm95"
)


# Compute classification accuracy using the independent
# validation samples.
#
# Replace 'GEE75MLP25' with any classification cube
# loaded from the mosaic directory to evaluate another model.

area_acc <- sits_accuracy(GEE75MLP25,
                          validation = valid_samples,
                          multicores = multicores
)
# Display the accuracy statistics
area_acc

# Display the confusion matrix
area_acc$error_matrix