# ==========================================================
# Multi-Source Land Cover Classification Pipeline using sits
# ==========================================================

# This script follows the cube generation workflow (1_CubeCreation.R) and:

# 1. Loads regularized data cubes
# 2. Merges EO datasets
# 3. Prepares training samples
# 4. Trains Random Forest & Produces probability cubes and classified maps
# 5. Trains TempCNN models & Produces probability cubes and classified maps

# ==========================================================

# ----------------------------------------------------------
# 0. LOAD REQUIRED PACKAGES
# ----------------------------------------------------------

#install.packages("torch")
library(torch)
#install.packages("sits")
library(sits)
#install.packages("sf")
library(sf)
#install.packages("stars")
library(stars)
#install.packages("purrr")
library(purrr)
#install.packages("dplyr")
library(dplyr)
#install.packages("dtwclust")
library(dtwclust)
#install.packages("dendextend")
library(dendextend)


#===============================================================
# 0.1. USER-DEFINED PARAMETERS
#===============================================================

# Main project directory (containing the project outputs)
project_dir <- "/home/users/h/havinga/MapNorthMadagascar/"

# Directory containing the regularized data cubes
Cube_dir <- "/home/users/h/havinga/CarteNMada/" 

# Data directory
data_dir <- file.path(project_dir, "Data/")

# ROI shapefile
roi_path <- file.path(data_dir, "ROI/RoiReady1.shp")

# Training samples
training_samples_path <- file.path(
  data_dir,
  "TrainingSamples/newTS_brut_2026/TS_roi1_2026Ready.shp"
)

# Output directories
output_rds <- file.path(data_dir, "SaveRDS2026/")
output_maps <- file.path(data_dir, "Map2026/")

# Time range
start_date <- "2021-01-01"
end_date <- "2025-01-01"

# Computational parameters
multicores <- 4
memsize <- 64

# Create output directories if needed
dir.create(output_rds, recursive = TRUE, showWarnings = FALSE)
dir.create(output_maps, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------
# 0.2. IMPORT REGION OF INTEREST (ROI)
# ----------------------------------------------------------

roi <- st_read(roi_path)

# Visualize ROI
plot(roi)

# ----------------------------------------------------------
# 1. LOAD REGULARIZED DATA CUBES
# ----------------------------------------------------------
# Function for loading previously generated regularized data cubes

load_cube <- function(source,
                      collection,
                      directory,
                      subdir,
                      roi) {
  
  cube <- sits_cube(
    source = source,
    collection = collection,
    roi = roi,
    data_dir = file.path(directory, subdir)
  )
  # Verify that the loaded cube is regularized
  if (!sits:::.cube_is_regular(cube)) {
    warning(
      paste("Cube in directory", subdir, "is not regularized.")
    )
  }
  
  return(cube)
}

# Load data cubes already created in 1_CubeCreation.R
s2_reg      <- load_cube("DEAFRICA", "SENTINEL-2-L2A", Cube_dir, "S2", roi)
indices_reg <- load_cube("DEAFRICA", "SENTINEL-2-L2A", Cube_dir, "indices", roi)
s1_reg      <- load_cube("MPC", "SENTINEL-1-RTC", Cube_dir, "S1_roi1", roi)
DEM_reg     <- load_cube("MPC", "COP-DEM-GLO-30", Cube_dir, "DEM", roi)
Rain_reg    <- load_cube("DEAFRICA", "RAINFALL-CHIRPS-MONTHLY", Cube_dir, "Rain_roi1", roi)


#===============================================================
# 2. Merge the cubes 
#===============================================================

# Merge Sentinel-2 reflectance data with the derived spectral indices
cube_mergedS2<- sits_merge(s2_reg, indices_reg)

# Merge Sentinel-1 data.
# The resulting cube is used for the TempCNN model,
# which only uses Sentinel-1 and Sentinel-2 variables
cube_mergedS1S2<- sits_merge(cube_mergedS2, s1_reg)

# Add topographic information
cube_mergedS1S2topo<- sits_merge(cube_mergedS1S2,DEM_reg)

# Add rainfall data
# This complete cube is used for the Random Forest model
cube_mergedAll<- sits_merge(cube_mergedS1S2topo,Rain_reg)


# Verify that the merged cube remains regular
sits:::.cube_is_regular(cube_mergedAll)

# Display the available bands
sits_bands(cube_mergedAll)

#Save the merged cubes as an RDS file
saveRDS(cube_mergedAll, file = file.path(output_rds, "cube_all_RF_roi1.rds"))
saveRDS(cube_mergedS1S2, file = file.path(output_rds, "cube_s1s2_TempCNN_roi1.rds"))

# Reload if the cube has already been generated
cube_mergedAll<- readRDS(file.path(output_rds, "cube_all_RF_roi1.rds"))

#===============================================================
# 3. RANDOM FOREST PROBABILITY CUBE PIPLINE
#===============================================================

#------------------------------------------------------------
# 3.1 Load the training data
#------------------------------------------------------------

#Read the shapefile containing the training samples
TS <- st_read(training_samples_path)

#extract time series as tibbles
raw_samples <- sits_get_data(
  samples = TS,
  cube = cube_mergedAll,
  start_date = start_date,
  end_date = end_date,
  n_sam_pol = 30,
  multicores = multicores,
  progress = TRUE
)

#Save the extracted samples as an RDS file
saveRDS(raw_samples, file = file.path(output_rds, "1_raw_samples_RF.rds"))

# Reload if already available.
raw_samples<-readRDS(file.path(output_rds, "1_raw_samples_RF.rds"))

# Export the training samples for use in Google Earth Engine in a shape file
raw_samples_export <- raw_samples %>%
  select(-time_series)  # Remove only the problematic column

raw_samples_sf <- st_as_sf(
  raw_samples_export,
  coords = c("longitude", "latitude"),
  crs = 4326
)

# Optional quality check
#names(raw_samples_sf)
#summary(raw_samples_sf)

#write into a shapefile, change the path and name of the shapefile
st_write(raw_samples_sf, "Data/TrainingSamples/TS_brutGEE/raw_samples.shp", delete_dsn = TRUE)


#-------------------
# 3.1.1 Statistical assessment of the raw training data (Sample Quality Assessment)
#-------------------
# Several methods can be used to assess the quality of the training data.
# A comprehensive description is available in the sits book, which is highly recommended.

# Assess the quality of the training samples
rfor_validate <- sits_kfold_validate(
  samples = raw_samples,
  folds = 5,
  ml_method = sits_rfor(),
  multicores = multicores
)
rfor_validate

# Ensure reproducibility
set.seed(42)

# Cluster the time series using Self-Organizing Maps (SOM)
# Parameters selected empirically after preliminary experiments.
som_cluster <- sits_som_map(raw_samples,
                            grid_xdim = 20,
                            grid_ydim = 20,
                            alpha = 1.0,
                            distance = "dtw",
                            rlen = 20
)

# Save the SOM model as an RDS file
saveRDS(som_cluster, file.path(output_rds, "som_cluster_RF.rds"))


# Produce a tibble with a summary of the mixed labels
# providing an overview of class confusion
som_eval <- sits_som_evaluate_cluster(som_cluster)

# Display the proportion of mixed samples in each cluster
som_eval
plot(som_eval, legend=legends)

#-------------------
# 3.1.2 Improve the training samples
# Clean and balance the training samples
#-------------------

# Two approaches are applied to improve the training dataset:
# 1) Remove noisy samples.
# 2) Balance the number of samples across classes.

# The evaluation described in Section 3.1.1
# can be repeated to assess the effect
# of sample cleaning and balancing.

# 1) detect noisy samples
all_samples <- sits_som_clean_samples(
  som_map = som_cluster, 
  prior_threshold = 0.6,
  posterior_threshold = 0.6,
  keep = c("clean", "analyze", "remove"))

# Print the sample distribution based on evaluation
plot(all_samples)

# Remove the noisy samples
clean_samples <- sits_som_clean_samples(
  som_map = som_cluster, 
  prior_threshold = 0.6,
  posterior_threshold = 0.6,
  keep = c("clean", "analyze"))

# Display a summary of the cleaned samples
summary(clean_samples)
sits_view(clean_samples)
# Optional visualization
plot(clean_samples)

# Save the cleaned training samples as an RDS file
saveRDS(clean_samples, file = file.path(output_rds, "2_clean_samples_RF.rds"))

# Re-evaluate the cleaned training samples.
som_cluster <- sits_som_map(clean_samples,
                            grid_xdim = 20,
                            grid_ydim = 20,
                            alpha = 1.0,
                            distance = "dtw",
                            rlen = 20
)
# Produce a tibble with a summary of the mixed labels 
# providing an overview of class confusion

som_eval <- sits_som_evaluate_cluster(som_cluster)
# Display the proportion of mixed samples in each cluster
som_eval
plot(som_eval, legend=legends)

# The rfor_validate() function (Section 3.1.1) can be run again
# to compare the cleaned samples with the original training dataset.


# 2) Balance the training samples
# Class balancing is performed independently
# from sample cleaning so that both strategies
# can be evaluated separately.

# - using the original training samples (raw_samples);
balanced_samples <- sits_reduce_imbalance(
  samples = raw_samples,
  n_samples_over = 150,
  n_samples_under = 300,
  multicores = 8)
# Optional summary
summary(balanced_samples)

# - using the cleaned training samples (clean_samples).
clean_balanced_samples <- sits_reduce_imbalance(
  samples = clean_samples,
  n_samples_over = 150,
  n_samples_under = 300,
  multicores = 8)
# Optional summary
summary(clean_balanced_samples)

# Save the balanced training datasets as RDS files
saveRDS(balanced_samples, file =file.path(output_rds, "3_balanced_samplesRF_tot.rds"))
saveRDS(clean_balanced_samples, file =file.path(output_rds, "4_clean_balanced_samplesRF_tot.rds"))


#---------------------------------------------------------------
# 3.2 RANDOM FOREST MODEL TRAINING
#---------------------------------------------------------------

# Reload the training datasets saved in Section 3.1
sample_files <- list(
  raw_samples            = file.path(output_rds, "1_raw_samples_RF.rds"),
  clean_samples          = file.path(output_rds, "2_clean_samples_RF.rds"),
  balanced_samples       = file.path(output_rds, "3_balanced_samplesRF_tot.rds"),
  clean_balanced_samples = file.path(output_rds, "4_clean_balanced_samplesRF_tot.rds")
)
samples_list <- lapply(sample_files, readRDS)

# Remove unnecessary columns that may contain NA values and cause errors
cols_to_drop <- c("post_prob", "eval", "id_sample", "id_neuron", "polygon_id", "prior_prob", "cube") 
samples_list <- lapply(samples_list, function(df) { dplyr::select(df, -dplyr::any_of(cols_to_drop)) })

#---------------------------------------------------------------
# 3.2.1 Train Random Forest models
#---------------------------------------------------------------

train_rf_model <- function(samples, ntrees = 1000) {
  sits_train(samples = samples,
             ml_method = sits_rfor(num_trees = ntrees))
}

#--------------------------------------------------------------------
# 3.2.2 Train and save the Random Forest models

# Define the output file names for each trained model
output_files <- list(
  raw_samples      = file.path(output_rds,"1_modelRF_brut.rds"),
  clean_samples        = file.path(output_rds,"2_modelRF_newsamp.rds"),
  balanced_samples   = file.path(output_rds, "3_modelRF_balsamp.rds"),
  clean_balanced_samples = file.path(output_rds,"4_modelRF_newbal.rds")
)

# Train one Random Forest model for each training dataset
# and save the resulting models as RDS files
models <- Map(function(samples, outfile) {
  model <- train_rf_model(samples)
  saveRDS(model, outfile)
  message("Model saved: ", outfile)
  return(model)
}, samples_list, output_files)


#---------------------------------------------------------------
# 3.3 Generate the Random Forest probability cube
#---------------------------------------------------------------

# Each trained Random Forest model is applied to the data cube
# to generate a probability cube. For every pixel, the output
# contains the probability of belonging to each land-cover class.

# Function for classifying the data cube using a trained Random Forest model
run_modelRF <- function(model_name, model_file) {
  
  # Display the name of the model currently being processed
  message("Processing model : ", model_name)
  
  # Load the trained Random Forest model
  modelRF <- readRDS(model_file)
  
  # Classify the data cube and generate the probability cube
  proba <- sits_classify(
    data       = cube_mergedAll,
    ml_model   = modelRF,
    multicores = multicores,
    output_dir = output_maps,
    version    = paste0("Proba_RF_", model_name, "_Roi1")
  )
  
  # Save the probability cube
  outfile <- file.path(
    output_rds,paste0(model_name, "_Proba_Roi1.rds")
  )
  
  saveRDS(proba, outfile)
  
  # Display a confirmation message
  message("Probability cube saved: ", outfile)
  
  return(proba)
}

# Apply the classification function to each trained Random Forest model.
# Depending on the size of the study area and the available computing resources,
# this step may take a considerable amount of time.

Proba_RF_brut <- run_modelRF(
  "1_modelRF_brut",
  file.path(output_rds,"1_modelRF_brut.rds")
)

Proba_RF_newsamp <- run_modelRF(
  "2_modelRF_newsamp",
  file.path(output_rds,"2_modelRF_newsamp.rds")
)

Proba_RF_balsamp <- run_modelRF(
  "3_modelRF_balsamp",
  file.path(output_rds,"3_modelRF_balsamp.rds")
)

Proba_RF_newbal <- run_modelRF(
  "4_modelRF_newbal",
  file.path(output_rds,"4_modelRF_newbal.rds")
)

# Reload previously generated probability cubes.
# If the cubes have already been created, this avoids rerunning
# the classification step above.

Proba_RF_brut    <- readRDS(file.path(output_rds,"1_modelRF_brut_Proba_Roi1.rds"))
Proba_RF_newsamp <- readRDS(file.path(output_rds,"2_modelRF_newsamp_Proba_Roi1.rds"))
Proba_RF_balsamp <- readRDS(file.path(output_rds,"3_modelRF_balsamp_Proba_Roi1.rds"))
Proba_RF_newbal  <- readRDS(file.path(output_rds,"4_modelRF_newbal_Proba_Roi1.rds"))

#---------------------------------------------------------------
#   RANDOM FOREST CLASSIFICATION
#---------------------------------------------------------------
# Several classification strategies can be considered:
# - classify the probability cube directly without variance estimation or smoothing;
# - apply a common smoothing parameter without estimating local variance;
# - estimate the logit variance first in order to define class-specific
#   smoothing parameters (recommended).

#---------------------------------------------------------------
# 3.4.1 Classification without spatial smoothing
#---------------------------------------------------------------
# Example using the model trained with the raw samples.
# However, estimating the variance before smoothing is strongly recommended.

# Convert the probability cube into a thematic map
# by assigning each pixel to its most probable class.

class_cube_roi1_wtSmooth <- sits_label_classification(
  cube = Proba_RF_brut,
  multicores = multicores,
  memsize = 12,
  output_dir = output_maps,
  version = "1_no_smooth_rf2_brut_roi1"
)

# Save the classified map
saveRDS(class_cube_roi1_wtSmooth, file = file.path(output_rds,"1_no_smooth_rf2_raw_roi1.rds"))

# Optional visualization of the classified map
plot(class_cube_roi1_wtSmooth,tile="39LUF",
     legend_text_size = 0.8, legend_position = "outside",legend=legends, title = "1_Roi1_RF_no_smooth"
)
# Optional interactive visualization
#sits_view(class_cube_roi1_wtSmooth, legend=legends, tiles ="39LUF")


#--------------------------------------------------------------------
# 3.4.2 Spatial smoothing based on logit variance estimates
#--------------------------------------------------------------------

# The variance estimation procedure can be applied to any
# probability cube. Here it is illustrated using one model
# (Proba_RF_newsamp) to derive the smoothing parameters.

# Estimate the logit variance for each land-cover class.
# These estimates are used to derive class-specific smoothing parameters.

Proba_cube_roi_var2<- sits_variance(
  cube = Proba_RF_newsamp,
  window_size = 7,
  neigh_fraction = 0.50,
  output_dir = output_maps, 
  multicores = multicores,
  memsize = 16)

# Save the variance cube
#saveRDS(Proba_cube_roi_var, file = file.path(output_rds,"1_Proba_RF2_brut_Roi1_var.rds"))
saveRDS(Proba_cube_roi_var2, file = file.path(output_rds,"2_Proba_RF2_newsamp_Roi1_var.rds"))
#saveRDS(Proba_cube_roi_var3, file = file.path(output_rds,"3_Proba_RF2_bal_Roi1_var.rds"))
#saveRDS(Proba_cube_roi_var4, file = file.path(output_rds,"4_Proba_RF_newbal_Roi1_var.rds"))

# Inspect the estimated logit variance distribution.
# The selected percentile is then used to define
# the class-specific smoothing parameters.
summary(Proba_cube_roi_var2)

# Example smoothing parameters derived from the
# selected logit variance percentile.
# These values should be recomputed for each study area.
smoothness <- c(
  "1011" = 1.44,
  "1012" = 1.49,
  "102" = 1.29,
  "103" = 1.4,
  "104" = 1.87,
  "105" = 1.67,
  "1131" = 1.39,
  "1132" = 1.39,
  "1133" = 1.25,
  "211" = 1.41,
  "221" = 1.37,
  "231" = 1.44,
  "232" = 1.71,
  "311" = 1.47,
  "312" = 1.45,
  "313" = 0.61,
  "321" = 1.72,
  "322" = 1.40)

# Reload the probability cubes.
# If they are already available in the current R session,
# this step can be skipped.
Proba_RF_brut <- readRDS(paste0(output_rds,"1_Proba_RF2_brut_Roi1.rds"))
Proba_RF_newsamp <- readRDS(paste0(output_rds,"2_modelRF_newsamp_Proba_Roi1.rds"))
Proba_RF_balsamp <- readRDS(paste0(output_rds,"3_Proba_RF2_bal_Roi1.rds"))
Proba_RF_newbal <- readRDS(paste0(output_rds,"4_modelRF_newbal_Proba_RF_Roi1.rds"))

# Apply Bayesian smoothing using the parameters derived
# from the 95th percentile of the logit variance.
# The smoothing values should be updated according to the
# summary of the variance cube.

class_bayes <- sits_smooth(
  cube = Proba_RF_brut,
  smoothness = smoothness,
  multicores = multicores,
  output_dir = output_maps,
  version = "1_95%_rf_brut_roi1"  
)

class_bayes2 <- sits_smooth(
  cube = Proba_RF_newsamp,
  smoothness = smoothness,
  multicores = multicores,
  output_dir = output_maps,
  version = "2_95%_rf_newsamp_roi1"
)

class_bayes3 <- sits_smooth(
  cube = Proba_RF_balsamp,
  smoothness = smoothness,
  multicores = multicores,
  output_dir = output_maps,
  version = "3_95%_rf_balsamp_roi1"
)

class_bayes4 <- sits_smooth(
  cube = Proba_RF_newbal,
  smoothness = smoothness,
  multicores = multicores,
  output_dir = output_maps,
  version = "4_95%_rf_newbal_roi1"
)

# Optional visualization of the smoothed classification
plot(class_bayes,tile="39LUF",
     legend_text_size = 0.8, legend_position = "outside",legend=legends, title = "1_Roi1_RF_no_smooth"
)

# Save the final smoothed outputs
saveRDS(class_bayes, file = file.path(output_rds,"1_95%_rf_brut_roi1.rds"))
saveRDS(class_bayes2, file = file.path(output_rds,"2_95%_rf_newsamp_roi1.rds"))
saveRDS(class_bayes3, file = file.path(output_rds,"3_95%_rf_bal_roi1.rds"))
saveRDS(class_bayes4, file = file.path(output_rds,"4_95%_rf_newbal_roi1.rds"))

#--------------------------------------------------------------------
# 3.4.3 Generate the final thematic map
#--------------------------------------------------------------------

# Reload one of the smoothed probability cubes.
# Skip this step if the objects are already available in memory.
class_bayes <- readRDS(file.path(output_rds,"1_95%_rf_brut_roi1.rds"))
class_bayes2 <- readRDS(file.path(output_rds,"2_95%_rf_newsamp_roi1.rds"))
class_bayes3 <- readRDS(file.path(output_rds,"3_95%_rf_bal_roi1.rds"))
class_bayes4 <- readRDS(file.path(output_rds,"4_95%_rf_newbal_roi1.rds"))

# Convert each smoothed probability cube into
# a final land-cover map.
# Each pixel is assigned to the class with the
# highest posterior probability.
labels_raw <- sits_label_classification(
  cube = class_bayes,
  multicores = multicores,
  output_dir = output_maps,
  version =  "1-95%-rf-roi1"
)
labels_clean <- sits_label_classification(
  cube = class_bayes2,
  multicores = multicores,
  output_dir = output_maps,
  version =  "2-95%-rf-roi1"
)
labels_balanced <- sits_label_classification(
  cube = class_bayes3,
  multicores = multicores,
  output_dir = output_maps,
  version =  "3-95%-rf-roi1"
)
labels_clean_balanced <- sits_label_classification(
  cube = class_bayes4,
  multicores = multicores,
  output_dir = output_maps,
  version =  "4-95%-rf-roi1"
)

# Save the classified maps
saveRDS(labels_raw, file = file.path(output_rds,"1_Class_95%_rf_brut_roi1.rds"))
saveRDS(labels_clean, file = file.path(output_rds,"2_Class_95%_rf_newsamp_roi1.rds"))
saveRDS(labels_balanced, file = file.path(output_rds,"3_Class_95%_rf_bal_roi1.rds"))
saveRDS(labels_clean_balanced, file = file.path(output_rds,"4_Class_95%_rf_newbal_roi1.rds"))

# Reload the classified maps if they have already
# been generated in a previous session.
labels_raw <- readRDS(file.path(output_rds,"1_Class_95%_rf_brut_roi1.rds"))
labels_clean <- readRDS(file.path(output_rds,"2_Class_95%_rf_newsamp_roi1.rds"))
labels_balanced<-  readRDS(file.path(output_rds,"3_Class_95%_rf_bal_roi1.rds"))
labels_clean_balanced <-  readRDS(file.path(output_rds,"4_Class_95%_rf_newbal_roi1.rds"))

#===============================================================
# 4. TEMPORAL CONVOLUTIONAL NEURAL NETWORK PROBABILITY CUBE PIPLINE
#===============================================================

#---------------------------------------------------------------
# 4.1 Extract the training samples
#---------------------------------------------------------------

# Reload the data cube containing only the Sentinel-1 and Sentinel-2 variables
# used for the TempCNN model.
cube_mergedS1S2 <- readRDS(file.path(output_rds,"cube_s1s2_TempCNN_roi1.rds"))

# Display the temporal coverage and available bands
sits_timeline(cube_mergedS1S2)
sits_bands(cube_mergedS1S2)

# Read the training sample shapefile
TS <- st_read(training_samples_path)

# Extract the time series associated with each training sample
raw_samplesS1S2<-sits_get_data(samples = TS,
                               cube = cube_mergedS1S2,
                               multicores = multicores,
                               start_date = start_date,
                               end_date = end_date,
                               crs =  4326,
                               n_sam_pol = 30,
                               progress = TRUE 
)

# Save the extracted samples as an RDS file
saveRDS(raw_samplesS1S2, file = file.path(output_rds, "1_raw_samplesS1S2_DP.rds"))

#-----------------------------------------------------------------------
# 4.1.1 Sample quality assessment and improvement
#-----------------------------------------------------------------------

# Reload the extracted training samples
raw_samplesS1S2 <- readRDS(file.path(output_rds, "1_raw_samplesS1S2_DP.rds"))

# Optional: Visualize the extracted time series
plot(raw_samplesS1S2)

# Several methods can be used to assess the quality
# of the training data.
# A comprehensive description is available
# in the sits book.
rfor_validate <- sits_kfold_validate(
  samples = raw_samplesS1S2,
  folds = 5,
  ml_method = sits_rfor(),
  multicores = multicores
)
rfor_validate

# Cluster the time series using Self-Organizing Maps (SOM)
# Parameters selected empirically after preliminary experiments.
som_cluster <- sits_som_map(raw_samplesS1S2,
                            grid_xdim = 20,
                            grid_ydim = 20,
                            alpha = 1.0,
                            distance = "dtw",
                            rlen = 20
)
# Optional: visualize the SOM clusters
plot(som_cluster, legend_position = "outside", band ="NDVI",legend_text_size="0.5")

# Save the SOM model
saveRDS(som_cluster, file = file.path(output_rds,"som_cluster_DL_brut.rds"))
# Reload the SOM model if needed
som_cluster <- readRDS(file.path(output_rds,"som_cluster_DL_brut.rds"))

# Evaluate the class mixture within each SOM cluster
som_eval <- sits_som_evaluate_cluster(som_cluster)
# Display the class mixture statistics
som_eval

#Optional visualization
plot(som_eval, legend=legends)

# Detect noisy training samples
all_samples <- sits_som_clean_samples(
  som_map = som_cluster, 
  prior_threshold = 0.6,
  posterior_threshold = 0.6,
  keep = c("clean", "analyze", "remove"))

# Optional : display the sample distribution after evaluation
plot(all_samples)

# Remove the noisy samples
clean_samples <- sits_som_clean_samples(
  som_map = som_cluster, 
  prior_threshold = 0.6,
  posterior_threshold = 0.6,
  keep = c("clean", "analyze"))

# Optional :display the cleaned sample distribution
summary(clean_samples)

# Save the cleaned training samples
saveRDS(clean_samples, file = file.path(output_rds,"2_clean_samples_DP.rds"))
# Reload the cleaned training samples
clean_samples <- readRDS(file.path(output_rds,"2_clean_samples_DP.rds"))

# Reassess the quality of the cleaned samples
rfor_validate <- sits_kfold_validate(
  samples = clean_samples,
  folds = 5,
  ml_method = sits_rfor(),
  multicores = multicores
)
rfor_validate

# Generate a new SOM using the cleaned samples
# Parameters selected empirically after preliminary experiments.
new_cluster <- sits_som_map(
  data = clean_samples,
  grid_xdim = 15,
  grid_ydim = 15,
  alpha = 1.0,
  rlen = 20,
  distance = "dtw")

# Evaluate the class mixture in the new SOM clusters
new_cluster_mixture <- sits_som_evaluate_cluster(new_cluster)
new_cluster_mixture

# Optional: visualize the updated SOM evaluation
plot(new_cluster_mixture)

# Balance the original training samples
balanced_samples <- sits_reduce_imbalance(
  samples = raw_samplesS1S2,
  n_samples_over = 120,
  n_samples_under = 350,
  multicores = multicores)

#Optional summary
summary(balanced_samples)

# Save the balanced training samples
saveRDS(balanced_samples, file = file.path(output_rds,"3_balanced_samples_DP.rds"))

# Balance the cleaned training samples
clean_balanced_samples <- sits_reduce_imbalance(
  samples = clean_samples,
  n_samples_over = 120,
  n_samples_under = 350,
  multicores = multicores)
summary(clean_balanced_samples)

# Save the cleaned and balanced training samples
saveRDS(clean_balanced_samples, file = file.path(output_rds,"4_clean_balanced_samples_DP.rds"))

# Reload if needed the balanced datasets
balanced_samples <- readRDS(file.path(output_rds,"3_balanced_samples_DP.rds"))
clean_balanced_samples<- readRDS(file.path(output_rds,"4_clean_balanced_samples_DP.rds"))

# Evaluate the final balanced training dataset
rfor_validate <- sits_kfold_validate(
  samples = clean_balanced_samples,
  folds = 5,
  ml_method = sits_rfor(),
  multicores = multicores
)
rfor_validate

#--------------------------------------------------------------
# 4.2 DEEP LEARNING MODEL TRAINING
#---------------------------------------------------------------

# The cleaned training dataset was selected
# for the remainder of the TempCNN workflow
# because it achieved the best validation
# performance during the sample quality assessment.

library(torch)
library(luz)

# Remove columns that may contain NA values
# and could interfere with model training.
cols_to_drop <- c(
  "post_prob", "eval", "id_sample",
  "id_neuron", "polygon_id", "prior_prob", "cube", "count"
)

clean_samples <- clean_samples %>%
  dplyr::select(-dplyr::any_of(cols_to_drop))

# Inspect the training dataset
View(clean_samples)

#---------------------------------------------------------------
# 4.2.1 Hyperparameter tuning of the TempCNN model
#---------------------------------------------------------------
set.seed(42)

# Perform hyperparameter optimization to identify
# the best TempCNN architecture and training parameters.

tuned_tempcnn <- sits_tuning(
  samples = clean_samples,
  ml_method = sits_tempcnn(),
  params = sits_tuning_hparams(
    cnn_layers = choice(c(256, 256, 256), c(128, 128, 128), c(64, 64, 64)),
    cnn_kernels = choice(c(3, 3, 3), c(5, 5, 5), c(7, 7, 7)),
    cnn_dropout_rates = choice(
      c(0.15, 0.15, 0.15), c(0.2, 0.2, 0.2),
      c(0.3, 0.3, 0.3), c(0.4, 0.4, 0.4)
    ),
    optimizer = torch::optim_adamw,
    opt_hparams = list(
      lr = loguniform(10^-3, 10^-4),
      weight_decay = loguniform(10^-2, 10^-8)
    )
  ),
  trials = 10,
  multicores = multicores
)

# Save and reload the tuning results
saveRDS(tuned_tempcnn, file = file.path(output_rds,"2_clean_tuned_tempcnn.rds"))
#load the saved RDS
tuned_tempcnn <- readRDS (file.path(output_rds,"2_clean_tuned_tempcnn.rds"))

# Display the tuning results
tuned_tempcnn

# Extract the hyperparameters of the best-performing model.
# Obtain accuracy, kappa, cnn_layers, cnn_kernels, and cnn_dropout_rates the best result
cnn_params <- tuned_tempcnn[1,c("accuracy", "kappa", "cnn_layers", "cnn_kernels", "cnn_dropout_rates"), ]
print(cnn_params)

# Extract the optimal learning rate and weight decay
hparams_best <- tuned_tempcnn[1,]$opt_hparams[[1]]

# Extract learning rate and weight decay 
lr_wd <- tibble::tibble(lr_best = hparams_best$lr, 
                        wd_best = hparams_best$weight_decay)
dplyr::bind_cols(cnn_params, lr_wd)


#---------------------------------------------------------------
# 4.2.2 Train the TempCNN model
#---------------------------------------------------------------
tempcnn_model <- sits_train(
  clean_samples,
  sits_tempcnn(
    optimizer            = torch::optim_adamw,
    cnn_layers           = c(256, 256, 256),
    cnn_kernels          = c(7,7,7),
    cnn_dropout_rates    = c(0.4,0.4, 0.4),
    epochs               = 20,
    batch_size           = 32,
    validation_split     = 0.1,
    verbose              = TRUE,
    opt_hparams = list(lr =  0.000892, weight_decay = 0.00158),
  )
)
# Optional: visualize the training history
plot(tempcnn_model)

# Save and reload the trained model
saveRDS(tempcnn_model, file = file.path(output_rds,"2_clean_samples_tempcnn_model.rds"))
tempcnn_model<- readRDS(file.path(output_rds,"2_clean_samples_tempcnn_model.rds"))

#--------------------------------------------------------------------
# 4.3 Generate the TempCNN probability cube
#--------------------------------------------------------------------
# Apply the trained TempCNN model to generate
# the probability cube.
ProbaCubeRoi1TempCnn<- sits_classify(
  data = cube_mergedS1S2,
  ml_model = tempcnn_model,
  output_dir = output_maps,
  multicores = multicores, 
  memsize = 64,
  verbose = TRUE,
  version = "Proba_TempCnnRoi1cleansamples"
)

# Save the probability cube
saveRDS(ProbaCubeRoi1TempCnn, file = file.path(output_rds,"2_ProbaCubeRoi1TempCnn.rds"))
# Reload the probability cube if it has already been generated
ProbaCubeRoi1TempCnn<-readRDS(file.path(output_rds,"2_ProbaCubeRoi1TempCnn.rds"))

#--------------------------------------------------------------------
# 4.4 Apply spatial smoothing
#--------------------------------------------------------------------
# 4.4.1 Estimate the logit variance

# Estimate the class-specific logit variance,
# which will be used to define the smoothing parameters.
Roi1TempCnn_var <- sits_variance(
  cube = ProbaCubeRoi1TempCnn,
  window_size = 7,
  neigh_fraction = 0.50,
  output_dir = output_maps, 
  multicores = multicores,
  memsize = 64,
  version = "2_Roi1TempCnn_var"
)

# Save and reload the variance estimates
saveRDS(Roi1TempCnn_var, file = file.path(output_rds,"2_Roi1TempCnn_var.rds"))
Roi1TempCnn_var<- readRDS(file.path(output_rds,"2_Roi1TempCnn_var.rds"))

# Inspect the variance distribution.
summary(Roi1TempCnn_var)

#----------------------------------------------------------
# 4.4.2 Spatial smoothing

# Apply Bayesian smoothing using the smoothing parameters
# derived from the selected variance percentile.
# Spatial smoothing reduces the salt-and-pepper effect
# and improves the classification of mixed pixels.
TempCNN_smooth <- sits_smooth(
  ProbaCubeRoi1TempCnn,
  output_dir = output_maps,
  version = "2-tcnn-smooth95",
  smoothness = c(
    "1011" = 1.28,
    "1012" = 1.31,
    "102" = 1.67,
    "103" = 2.78,
    "104" = 0.83,
    "105" = 1.04,
    "1131" = 1.86,
    "1132" = 3.27,
    "1133" = 3.42,
    "211" = 1.96,
    "221" = 4.04,
    "231" = 4.16,
    "232" = 2.99,
    "311" = 1.33,
    "312" = 1.34,
    "313"= 2.08,
    "321" = 1.26,
    "322" = 2.40),
  multicores = multicores,  
  memsize = 64
)

# Save the smoothed probability cube for future reuse.
saveRDS(TempCNN_smooth, file = file.path(output_rds,"2_Proba_smooth95_dp.rds"))
# Reload the smoothed probability cube
TempCNN_smooth <- readRDS(file.path(output_rds,"2_Proba_smooth95_dp.rds"))
#--------------------------------------------------------------------
# 4.5 Generate the final thematic map
#--------------------------------------------------------------------

# Convert the smoothed probability cube into
# the final land-cover classification map.

# Obtain the final labelled map (thematic map)
labels<- sits_label_classification(
  TempCNN_smooth,
  output_dir = output_maps,
  version = "2-tempcnn-map-95",
  multicores = multicores,
  memsize = 64
)

# Optional: display the classified map
plot(labels, legend=legends, legend_position = "outside", tile ="39LUF")

# Optionnal : interactive visualization
sits_view(labels, legend= legends, tiles = "39LUF", "39LUG")
# Save the classified map
saveRDS(labels, file = file.path(output_rds,"2_class_smooth95_dp.rds"))
# Reload the classified map if it has already been generated
labels <- readRDS(file.path(output_rds,"2_class_smooth95_dp.rds"))

