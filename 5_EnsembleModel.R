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
# 0. Load input data
# 1. Define the ensemble prediction function
# 2. Load the probability cubes
# 3. Generate ensemble classification maps
# 4. Crop classification maps to the Region of Interest
# 5. Generate validation samples
# 6. Prepare the validation dataset
# 7. Accuracy assessment
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

# Load the Region of Interest (ROI).
# Update the shapefile path if using a different study area.
roi <- st_read(paste0(directoryData, "/ROI/RoiReady1.shp"))

plot(roi)

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
    memsize = 64,
    multicores = 8
  )
  # Apply Bayesian spatial smoothing.
  # The smoothing parameters were previously estimated
  # from the logit variance analysis (see Script 2_MapCreation).
  patchworksmoothparam <- sits_smooth(
    patchwork,
    output_dir = paste0(directory,"/EnsembleModel"),
    version = paste0(version,"sm95"),
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
    multicores = 16,  
    memsize = 64
  )
  # Convert the smoothed probability cube
  # into the final land-cover classification map.
  labels <- sits_label_classification(
    cube = patchworksmoothparam,
    multicores = 4,
    memsize = 16,
    output_dir = paste0(directory,"/EnsembleModel"),
    version = paste0(version,"sm95"),
    threshold = 0.1
  )
  # Save the classified map
  saveRDS(labels, file = paste0(directoryData,"/SaveRDS2026/EnsembleModel/", version,"sm95.rds"))
  return(labels)
}

#--------------------------------------------------------------
# 2. Load the probability cubes
#--------------------------------------------------------------

# Load the probability cube generated in
# Google Earth Engine.
GEE_probs <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = "Data/Roi1ProbaGEENew/ProcessedGEETile",
  bands = "probs",
  labels = label,
  roi=roi, 
  parse_info = c("satellite", "sensor", "tile", 
                 "start_date", "end_date", "band", "version"))

# Generate the corresponding thematic map
GEE_map <- sits_label_classification(
  cube = GEE_probs,
  output_dir = paste0(directoryData, "/Map"),
  version = "GEE_4years",
  memsize = 128,
  multicores = 8
)

list.files("Data/Roi1ProbaGEE4dates/CubeGEE/CalibCubeGEE")


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



# Load the Random Forest probability cube.
RF_probs <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData, "/Map2026/RandomForest-3bal"),
  bands = "probs",
  labels = label,
  roi=roi, 
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
  cubes = list(GEE_probs, TempCNN_probs, RF_probs), 
  version = "A-34GEE-33Temp-33RF", 
  weights= c(0.34 , 0.33, 0.33),
  directory = paste0(directoryData, "/Map2026")
)

# Example: 50% Google Earth Engine,
# 25% TempCNN, and 25% Random Forest.
labels2 <- EnsembleModel(
  cubes = list(GEE_probs, TempCNN_probs, RF_probs), 
  version = "A-50GEE-25Temp-25RF", 
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
labels10 <- EnsembleModel(
  cubes = list(GEE_probs, RF_probs), 
  version = "A-75GEE-25RF", 
  weights= c(0.75 , 0.25),
  directory = paste0(directoryData, "/Map2026")
)

#----------------------------------------------------------
# Reload previously generated ensemble maps
#----------------------------------------------------------
RDSdirect <- "/home/users/h/havinga/MapNorthMadagascar/Data/SaveRDS2026/EnsembleModel/"
# Naming convention:
# A = weighted average
# followed by the contribution of each model.

labels <- readRDS(paste0(RDSdirect, "A-34GEE-33Temp-33RFsm95.rds"))
labels2 <- readRDS(paste0(RDSdirect, "A-50GEE-25Temp-25RFsm95.rds"))
labels3 <- readRDS(paste0(RDSdirect, "A-25GEE-75Temp95sm95.rds"))
labels4 <- readRDS(paste0(RDSdirect,"A-50GEE-50Temp95sm95.rds"))
labels5 <- readRDS(paste0(RDSdirect,"A-75GEE-25Temp95sm95.rds"))
labels6 <- readRDS(paste0(RDSdirect,"A-63GEE-37Temp95sm95.rds"))
labels7 <- readRDS(paste0(RDSdirect,"A-87GEE-13Temp95sm95.rds"))
labels8 <- readRDS(paste0(RDSdirect,"A-34GEE-33Temp-33RFsm95.rds"))
labels9 <- readRDS(paste0(RDSdirect,"A-50GEE-25Temp-25RFsm95.rds"))

gee<- readRDS(paste0(RDSdirect,"GEEsm95.rds"))

# Color palette used for visualizing
# the land-cover classes.
legends = c(
  "1011" = "#4CAF50",   # vert profond, forêt humide basse altitude
  "1012" = "#1B5E20",   # vert sombre, forêt humide moyenne altitude
  "102"  = "#A3A142",   # vert mousse, forêt humide semi-décidue
  "103"  = "#A67B5B",   # brun chaud, forêt dense sèche
  "104"  = "#AD5D76",   # rose-magenta adouci, mangroves
  "105"  = "#6A5ACD",   # violet soutenu, forêt littorale
  "1131" = "#5C8649",   # vert vif, forêt secondaire humide
  "1132" = "#D6CB5B",   # jaune olive, forêt secondaire sèche
  "1133" = "#9BCB7D",   # vert clair, cordon forestier
  "211"  = "#ADD8E6",   # bleu pâle, fourré littoral
  "221"  = "#FFFFBE",   # jaune très clair, mosaïques herbeuses boisées
  "311"  = "#1565C0",   # bleu profond, eau
  "312"  = "#90CAF9",   # bleu clair, zones humides
  "321"  = "#E3E3E3",   # beige ocre, sol nu
  "322"  = "#424242",   # gris anthracite, urbain
  "231"  = "#B74F4F",   # brun-rouge, culture
  "313"  = "#B74F4F",   # rouge terre, culture humide
  "232"  = "#F6D55C"    # jaune doré doux, prairie / pâturage sec
)
# Display the classified map
plot(labels4, tile = "39LTG", legend=legends, legend_text_size = 0.8, legend_position = "outside", 
     title = "Roi1_RF_brut"
)
# Interactive visualization
sits_view(labels5, legends=legends)



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
  pattern = "class_3-95%-rf-roi1.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/RF_3"),
  roi = roi
)
#Gee34-TempCNN33-RF33
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-34gee-33temp-33rfsm95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee34-TempCNN33-RF33"),
  roi = roi
)
#Gee50-TempCNN25-RF25
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-50gee-25temp-25rfsm95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee50-TempCNN25-RF25"),
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

#Gee75-RF25
crop_rasters_roi(
  path = file.path(directoryData, "Map2026/EnsembleModel"),
  pattern = "class_a-75gee-25rfsm95.*\\.tif$",
  outputdir = file.path(directoryData, "Map2026/Crop_Roi1/Gee75-RF25"),
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
  multicores = 1
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
valid_samples<- st_read(paste0(directoryData,"/TrainingSamples/validSamplesChecked/ValidSamplesChecked.shp"))

# List of valid land-cover labels
valid_label <- c(
  "1011","1012","102","103","104","105",
  "1131","1132","1133","211","221","231","232",
  "311","312","313","321","322"
)

# Keep only valid samples and update
# the reference labels when required.
valid_samples <- valid_samples %>%
  filter(
    is.na(Valid) |
      Valid == "" |
      Valid %in% valid_label
  ) %>%
  mutate(
    label = ifelse(
      !is.na(Valid) & Valid != "" & Valid %in% valid_label,
      Valid,
      label
    )
  )

# Display a summary of the cleaned dataset
summary(valid_samples)
# Save the cleaned validation samples
saveRDS(valid_samples, file = paste0(RDSdirect, "valid_samples_clean.rds"))
# Reload the cleaned validation samples
valid_samples<- readRDS(paste0(RDSdirect, "valid_samples_clean.rds"))

#----------------------------------------------------------
# 7. Accuracy assessment
#----------------------------------------------------------

# Load the classified maps to be evaluated.

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
plot(TempCNN)

RF <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/RF_3"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "3-95%-rf-roi1"
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

Gee34 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee34-TempCNN33-RF33"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-34gee-33temp-33rfsm95"
)

Gee50 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee50-TempCNN25-RF25"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-50gee-25temp-25rfsm95"
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

Gee50TempCNN50 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee50-TempCNN50"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-50gee-50temp95sm95"
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


Gee87TempCNN13 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee87-TempCNN13"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-87gee-13temp95sm95"
)

Gee75RF25 <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Crop_Roi1/Gee75-RF25"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "a-75gee-25rfsm95"
)

Geedirect <- sits_cube(
  source = "DEAFRICA",
  collection = "SENTINEL-2-L2A",
  data_dir = paste0(directoryData,"/Map2026/Geedirect2"),
  bands = "class",
  roi = roi,
  labels = label,
  version = "MULTI"
)
plot(Geedirect)

# Compute the classification accuracy
# using the independent validation samples.

# Replace 'GEE' with any of the classification maps
# loaded above (e.g., TempCNN, RF, Gee34, Gee50,
# Gee25TempCNN75, Gee50TempCNN50, Gee75TempCNN25,
# Gee87TempCNN13, Gee75RF25, or Geedirect) to
# evaluate the corresponding model.
area_acc <- sits_accuracy(GEE,
                          validation = valid_samples,
                          multicores = 8
)
# Display the accuracy statistics
area_acc

# Display the confusion matrix
area_acc$error_matrix
