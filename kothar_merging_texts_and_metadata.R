# Load relevant libraries
library(dplyr)
library(jsonlite)
library(readr)
library(openxlsx)
library(purrr)

# Load datasets
json_file <- fromJSON("#Add your file path here")
json_cases <- as.data.frame(json_file)

cases_file <- read_csv("#Add your file path here")
csv_cases <- as.data.frame(cases_file)

# Format json_cases before merge
json_cases <- json_cases %>%
  rename(case_id = cluster_id) %>%
  select(-opinions)

# Find and remove duplicates
table(duplicated(csv_cases$case_id))  # Check for duplicates in csv_cases
table(duplicated(json_cases$case_id)) # Check for duplicates in json_cases

csv_cases <- csv_cases[!duplicated(csv_cases$case_id), ]
json_cases <- json_cases[!duplicated(json_cases$case_id), ]

# Format columns before merging
json_cases$case_id <- as.character(unlist(json_cases$case_id))
csv_cases$case_id <- as.character(csv_cases$case_id)

# Merge dataframes
merged_df <- full_join(csv_cases, json_cases, by = "case_id")

# Find missing cases
missing_case <- merged_df %>%
  filter(is.na(opinion_text))

# Format merged_df
merged_df <- merged_df %>%
  select(-meta)

# Define safe converter
safe_to_string <- function(x) {
  if (is.null(x)) return(NA_character_)
  if (is.atomic(x)) return(toString(x))
  if (is.list(x)) return(toString(unlist(x, recursive = TRUE)))
  return(as.character(x))
}

# Apply it only to the list/matrix columns
merged_df <- merged_df %>%
  mutate(across(all_of(list_col_names), ~ map_chr(., safe_to_string)))

# Convert more
cleaned_df <- merged_df

for (col in list_col_names) {
  message(glue::glue("Processing: {col}"))
  
  # Try to flatten using safe_to_string
  tryCatch({
    cleaned_df[[col]] <- map_chr(cleaned_df[[col]], function(x) {
      if (is.null(x)) return(NA_character_)
      if (is.atomic(x)) return(toString(x))
      if (is.list(x)) return(toString(unlist(x, recursive = TRUE)))
      return(as.character(x))
    })
  }, error = function(e) {
    warning(glue::glue("Skipping column {col}: {e$message}"))
  })
}

# Save merged and cleaned data
cleaned_df <- cleaned_df %>% select(-dateReargued)
write_csv(cleaned_df, "#Add your file path here")



