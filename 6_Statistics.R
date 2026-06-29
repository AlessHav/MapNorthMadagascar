# =========================================================
# MAP COMPARISON USING MCNEMAR TEST
# =========================================================
# This script statistically compares multiple land-cover
# classification maps using the McNemar test.
#
# Main steps:
# 1. Load the Region of Interest (ROI) and validation points
# 2. Extract predicted classes from each map
# 3. Compare classification results using McNemar's test
# 4. Correct p-values using the Holm method
# 5. Export statistical comparison results
# =========================================================

library(terra)
library(dplyr)
library(sits)
library(sf)

# =========================================================
# 1. Paths
# =========================================================
# Update these paths to match your local project structure.
directoryData <- "/home/users/h/havinga/MapNorthMadagascar/Data"

# Load the Region of Interest (ROI).
# Update the shapefile path if using a different study area.
roi <- st_read(paste0(directoryData, "/ROI/RoiReady1.shp"))

# Display the Region of Interest.
plot(roi)

# =========================================================
# 2. LOAD VALIDATION POINTS
# =========================================================
# Load the cleaned validation dataset.
val_pts<- readRDS(paste0(directoryData,"/SaveRDS2026/EnsembleModel/valid_samples_clean.rds"))


# Convert the validation dataset to a
# SpatVector object for raster extraction.
val_pts<-vect(val_pts)
# Name of the reference label column.
truth_col <- "label"

# =========================================================
# 3. CLASS CODE MAPPING
# =========================================================
# Define the land-cover class codes.
class_codes <- c(
  "1011","1012","102","103","104","105",
  "1131","1132","1133","211","221","231","232",
  "311","312","313","321","322"
)

# Assign integer identifiers.
names(class_codes) <- 1:18

# Create a lookup table converting land-cover
# class codes into the integer IDs stored
# in the classified raster maps.
class_to_id <- setNames(
  names(class_codes),
  class_codes
)

# =========================================================
# 4. MULTI-TILE EXTRACTION FUNCTION
# =========================================================
#----------------------------------------------------------
# Extract predicted classes from multiple raster tiles.
#
# Arguments:
#   files       : vector of classified raster files
#   val_pts     : validation points (SpatVector)
#   truth_col   : reference label column
#   class_to_id : lookup table converting class codes
#
# Returns:
#   A data frame containing the reference and predicted
#   classes for every validation point.
#----------------------------------------------------------
extract_multitile <- function(
    files,
    val_pts,
    truth_col,
    class_to_id
){
  
  # Initialize an empty list to store
  # the extracted predictions.
  extracted_list <- list()
  
  # Process each raster tile independently.
  for(i in seq_along(files)){
    
    cat(
      "Processing:",
      basename(files[i]),
      "\n"
    )
    # Load the current raster tile.
    r <- rast(files[i])
    
    # Reproject the validation points
    # to the raster coordinate reference system.
    pts_proj <- project(val_pts, r)
    
    # Extract the predicted class at each
    # validation point.
    ext <- terra::extract(
      r,
      pts_proj
    )
    # Build a data frame containing
    # reference and predicted labels.
    tmp <- data.frame(
      point_id = seq_len(nrow(val_pts)),
      truth = as.character(
        values(val_pts)[[truth_col]]
      ),
      pred = as.character(ext[,2])
    )
    
    # Convert the reference labels to the
    # integer IDs used by the raster
    tmp$truth <- class_to_id[tmp$truth]
    
    # Remove validation points not covered
    # by the current raster tile.
    tmp <- tmp[!is.na(tmp$pred), ]
    
    extracted_list[[i]] <- tmp
  }
  # Merge all extracted data frames.
  result <- bind_rows(extracted_list)
  
  # Keep only one prediction per validation
  # point in case of tile overlap.
  
  result <- result %>%
    group_by(point_id) %>%
    slice(1) %>%
    ungroup()
  
  return(result)
}

# =========================================================
# 5. LOAD CLASSIFICATION MAPS
# =========================================================

# Locate the raster files corresponding
# to the ensemble classification.
ensemble_files <- list.files(
  paste0(directoryData,
         "/Map2026/Crop_Roi1/Gee75-TempCNN25"),
  pattern = "class_a-75gee-25temp.*\\.tif$",
  full.names = TRUE
)
# Locate the raster files corresponding
# to the Geo-Embedding classification.
geo_files <- list.files(
  paste0(directoryData,
         "/Map2026/Crop_Roi1/Gee-sits"),
  pattern = "class_gee.*\\.tif$",
  full.names = TRUE
)

# Example: load additional maps
# (e.g. TempCNN) if needed.

# tempcnn_files <- list.files(
#   paste0(directoryData, "/Map/TempCNN"),
#   pattern = "\\.tif$",
#   full.names = TRUE
# )

# =========================================================
# 6. EXTRACT PREDICTED CLASSES
# =========================================================

# Extract predictions from the
# ensemble classification.
EnsembleCNN_df <- extract_multitile(
  files = ensemble_files,
  val_pts = val_pts,
  truth_col = truth_col,
  class_to_id = class_to_id
)

# Extract predictions from the
# Geo-Embedding classification.
Geo_df <- extract_multitile(
  files = geo_files,
  val_pts = val_pts,
  truth_col = truth_col,
  class_to_id = class_to_id
)

# Exemple :

# TempCNN_df <- extract_multitile(
#   files = tempcnn_files,
#   val_pts = val_pts,
#   truth_col = truth_col,
#   class_to_id = class_to_id
# )

# =========================================================
# 7. MCNEMAR TEST FUNCTION
# =========================================================

#----------------------------------------------------------
# Compare two classification maps using
# McNemar's statistical test.
#
# Arguments:
#   df1   : predictions from the first map
#   df2   : predictions from the second map
#   name1 : first map name
#   name2 : second map name
#
# Returns:
#   A data frame containing the McNemar
#   statistic and p-value.
#----------------------------------------------------------
run_mcnemar <- function(
    df1,
    df2,
    name1,
    name2
){
  # Match predictions from both maps
  # using the validation point IDs.
  comparison_df <- df1 %>%
    select(
      point_id,
      truth,
      pred
    ) %>%
    rename(
      pred1 = pred
    ) %>%
    inner_join(
      df2 %>%
        select(
          point_id,
          pred
        ) %>%
        rename(
          pred2 = pred
        ),
      by = "point_id"
    )
  # Determine whether each map correctly
  # classified each validation point.
  correct1 <- (
    comparison_df$truth ==
      comparison_df$pred1
  )
  
  correct2 <- (
    comparison_df$truth ==
      comparison_df$pred2
  )
  # Build the McNemar contingency table.
  tab <- table(
    correct1,
    correct2
  )
  # Perform McNemar's test.
  test <- mcnemar.test(tab)
  # Return the statistical summary.
  data.frame(
    map1 = name1,
    map2 = name2,
    n_points = nrow(comparison_df),
    statistic = as.numeric(
      test$statistic
    ),
    p_value = test$p.value
  )
}

# =========================================================
# 8. DEFINE MAPS TO COMPARE
# =========================================================
# Create a named list containing all
# classification maps to be compared.
maps <- list(
  EnsembleCNN = EnsembleCNN_df,
  GeoEmbedding = Geo_df
  
  # TempCNN = TempCNN_df
)

# =========================================================
# 9. RUN ALL PAIRWISE COMPARISONS
# =========================================================
# Generate all possible pairwise
# combinations of classification maps.
pairs <- combn(
  names(maps),
  2,
  simplify = FALSE
)
# Apply McNemar's test to every
# pair of classification maps.
results <- lapply(
  pairs,
  function(x){
    
    run_mcnemar(
      df1 = maps[[x[1]]],
      df2 = maps[[x[2]]],
      name1 = x[1],
      name2 = x[2]
    )
    
  }
)
# Merge all statistical results into
# a single data frame.
results <- bind_rows(results)

# =========================================================
# 10. HOLM MULTIPLE-TEST CORRECTION
# =========================================================

# Adjust the p-values for multiple
# comparisons using Holm's correction.
results$p_holm <- p.adjust(
  results$p_value,
  method = "holm"
)
# Identify statistically significant
# differences between maps.
results$significant <- ifelse(
  results$p_holm < 0.05,
  "YES",
  "NO"
)

# =========================================================
# 11. SORT RESULTS
# =========================================================
# Sort the comparisons according
# to the adjusted p-values.
results <- results %>%
  arrange(p_holm)

# Display the statistical summary.
print(results)

# =========================================================
# 12. EXPORT RESULTS
# =========================================================
# Export the statistical comparison
# results as a CSV file.
write.csv(
  results,
  "McNemar_Holm_results.csv",
  row.names = FALSE
)
